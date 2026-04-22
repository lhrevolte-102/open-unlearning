import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import datasets
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import RandomSampler, SequentialSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.qa import QADataset, QAwithAlternateDataset, QAwithIdkDataset
from data.utils import load_allowed_indices
from utils.runtime import configure_torch_checkpoint_safe_globals
from selective.runtime import apply_intra_stage_ordering
from selective.utils import (
    build_reference_manifest,
    build_reference_split_manifests,
    build_stage_manifests,
)
from selective.score import build_difficulty_payload, compute_unlearning_forget_losses
from selective.pipeline import prepare_difficulty_payload
from utils.tensorboard import get_tensorboard_log_dir, log_tensorboard_metrics
from trainer.base import FinetuneTrainer


def _fake_preprocess_chat_instance(
    tokenizer,
    template_config,
    prompt_msgs,
    response_msgs,
    max_length,
    predict_with_generate=False,
):
    if isinstance(response_msgs, list):
        response = response_msgs[-1]
    else:
        response = response_msgs
    token = sum(ord(char) for char in response) % 7
    return {
        "input_ids": torch.tensor([1, token + 2, token + 3]),
        "labels": torch.tensor([-100, token % 2, (token + 1) % 2]),
        "attention_mask": torch.tensor([1, 1, 1]),
    }


def _make_template_args():
    return {
        "apply_chat_template": False,
        "user_start_tag": "Q: ",
        "user_end_tag": "\n",
        "asst_start_tag": "A: ",
        "asst_end_tag": "\n",
    }


class _FakeModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self._logits = logits

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        batch_size = input_ids.shape[0]
        logits = self._logits[:batch_size].to(input_ids.device)
        return SimpleNamespace(logits=logits)


def _make_logits(preferred_tokens):
    logits = []
    for token_pair in preferred_tokens:
        sample_logits = []
        for token in token_pair:
            row = torch.full((2,), -4.0)
            row[token] = 4.0
            sample_logits.append(row)
        sample_logits.append(torch.zeros(2))
        logits.append(torch.stack(sample_logits))
    return torch.stack(logits)


class _FakeSummaryWriter:
    def __init__(self):
        self.scalars = []
        self.flushed = False

    def add_scalar(self, tag, value, global_step=None):
        self.scalars.append((tag, value, global_step))

    def flush(self):
        self.flushed = True


class TestSelective:
    def test_configure_torch_checkpoint_safe_globals_allows_numpy_rng_load(self):
        clear_safe_globals = getattr(torch.serialization, "clear_safe_globals", None)
        if clear_safe_globals is None:
            pytest.skip("torch.serialization.clear_safe_globals is unavailable")

        numpy_rng_payload = {"numpy": np.random.get_state()}

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "rng_state.pth"
            torch.save(numpy_rng_payload, checkpoint_path)

            clear_safe_globals()
            with pytest.raises(Exception):
                torch.load(checkpoint_path, weights_only=True)

            configure_torch_checkpoint_safe_globals()
            restored = torch.load(checkpoint_path, weights_only=True)

        assert "numpy" in restored

    def test_get_tensorboard_log_dir_uses_output_logs_subdirectory(self):
        assert get_tensorboard_log_dir("/tmp/selective-run") == Path(
            "/tmp/selective-run/logs"
        )

    def test_log_tensorboard_metrics_ignores_non_scalars(self):
        writer = _FakeSummaryWriter()

        log_tensorboard_metrics(
            writer,
            {
                "loss": torch.tensor(1.25),
                "accuracy": 0.5,
                "skipped_text": "not-a-scalar",
                "skipped_dict": {"nested": 1},
            },
            prefix="tofu",
            step=7,
        )

        assert writer.scalars == [
            ("tofu/loss", 1.25, 7),
            ("tofu/accuracy", 0.5, 7),
        ]
        assert writer.flushed

    def test_reference_manifest_contains_expected_records(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            models_dir = tmp_path / "models"
            for split_id in range(2):
                (models_dir / f"split{split_id}").mkdir(parents=True, exist_ok=True)

            split_manifests = build_reference_split_manifests(
                all_indices=[0, 1, 2, 3],
                output_dir=tmp_path / "folds",
                num_repeats=1,
                repeat_split_seed=0,
            )
            reference_manifest = build_reference_manifest(
                metadata={
                    "method": "NPO",
                    "model": "base-model",
                    "forget_split": "forget10",
                    "retain_split": "retain90",
                    "num_repeats": 1,
                    "repeat_split_seed": 0,
                    "all_indices": [0, 1, 2, 3],
                },
                reference_split_manifests=split_manifests,
                checkpoint_root_dir=models_dir,
                reference_manifest_output_path=tmp_path / "reference_models.json",
                validate_checkpoint_paths=True,
            )

            assert reference_manifest["metadata"]["num_reference_models"] == 2
            assert len(reference_manifest["references"]) == 2
            assert reference_manifest["references"][0]["checkpoint_path"].endswith(
                "split0"
            )

    def test_random_repeated_halving_manifests_cover_full_dataset_per_repeat(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            split_manifests = build_reference_split_manifests(
                all_indices=list(range(7)),
                output_dir=tmp_dir,
                num_repeats=3,
                repeat_split_seed=11,
            )

        assert len(split_manifests) == 6
        heldout_counts = {idx: 0 for idx in range(7)}
        for repeat_id in range(3):
            repeat_splits = [
                split for split in split_manifests if split["repeat_id"] == repeat_id
            ]
            assert len(repeat_splits) == 2

            part0 = next(split for split in repeat_splits if split["partition_id"] == 0)
            part1 = next(split for split in repeat_splits if split["partition_id"] == 1)

            assert part0["split_id"] == repeat_id * 2
            assert part1["split_id"] == repeat_id * 2 + 1
            assert part0["split_seed"] == 11 + repeat_id
            assert part1["split_seed"] == 11 + repeat_id
            assert set(part0["train_indices"]) | set(part0["heldout_indices"]) == set(
                range(7)
            )
            assert not (set(part0["train_indices"]) & set(part0["heldout_indices"]))
            assert part0["train_indices"] == part1["heldout_indices"]
            assert part1["train_indices"] == part0["heldout_indices"]

            for idx in part0["heldout_indices"]:
                heldout_counts[idx] += 1
            for idx in part1["heldout_indices"]:
                heldout_counts[idx] += 1

        assert all(count == 3 for count in heldout_counts.values())

    def test_random_repeated_halving_requires_explicit_repeat_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(
                ValueError, match="num_repeats must be explicitly set"
            ):
                build_reference_split_manifests(
                    all_indices=[0, 1, 2, 3],
                    output_dir=tmp_dir,
                    num_repeats=None,
                )

    def test_qa_dataset_filters_allowed_indices(self):
        dataset = datasets.Dataset.from_dict(
            {
                "question": ["q0", "q1", "q2"],
                "answer": ["a0", "a1", "a2"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "stage1.json"
            manifest_path.write_text(
                json.dumps({"allowed_indices": [0, 2]}), encoding="utf-8"
            )

            with patch("data.qa.load_hf_dataset", return_value=dataset), patch(
                "data.qa.preprocess_chat_instance", _fake_preprocess_chat_instance
            ):
                qa_dataset = QADataset(
                    hf_args={"path": "unused"},
                    template_args=_make_template_args(),
                    tokenizer=None,
                    allowed_indices_path=manifest_path,
                )

        assert len(qa_dataset) == 2
        assert qa_dataset.data["index"] == [0, 2]

    def test_load_allowed_indices_can_preserve_manifest_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "stage1.json"
            manifest_path.write_text(
                json.dumps({"allowed_indices": [2, 0, 2, 1]}), encoding="utf-8"
            )

            assert load_allowed_indices(manifest_path) == [0, 1, 2]
            assert load_allowed_indices(manifest_path, preserve_order=True) == [
                2,
                0,
                1,
            ]

    def test_qa_dataset_preserves_manifest_order_when_requested(self):
        dataset = datasets.Dataset.from_dict(
            {
                "question": ["q0", "q1", "q2"],
                "answer": ["a0", "a1", "a2"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "stage1.json"
            manifest_path.write_text(
                json.dumps({"allowed_indices": [2, 0, 2, 1]}), encoding="utf-8"
            )

            with patch("data.qa.load_hf_dataset", return_value=dataset), patch(
                "data.qa.preprocess_chat_instance", _fake_preprocess_chat_instance
            ):
                qa_dataset = QADataset(
                    hf_args={"path": "unused"},
                    template_args=_make_template_args(),
                    tokenizer=None,
                    allowed_indices_path=manifest_path,
                    preserve_manifest_order=True,
                )

        assert len(qa_dataset) == 3
        assert qa_dataset.data["index"] == [2, 0, 1]

    def test_idk_dataset_deterministic_sampling_and_index(self):
        dataset = datasets.Dataset.from_dict({"question": ["q0"], "answer": ["a0"]})
        with tempfile.TemporaryDirectory() as tmp_dir:
            idk_path = Path(tmp_dir) / "idk.jsonl"
            idk_path.write_text("idk-0\nidk-1\nidk-2\n", encoding="utf-8")

            with patch("data.qa.load_hf_dataset", return_value=dataset), patch(
                "data.qa.preprocess_chat_instance", _fake_preprocess_chat_instance
            ):
                qa_dataset = QAwithIdkDataset(
                    hf_args={"path": "unused"},
                    template_args=_make_template_args(),
                    tokenizer=None,
                    idk_path=str(idk_path),
                    idk_sampling_mode="deterministic",
                    idk_sampling_seed=5,
                )

                first = qa_dataset[0]
                second = qa_dataset[0]

        assert torch.equal(
            first["alternate"]["input_ids"], second["alternate"]["input_ids"]
        )
        assert first["alternate"]["index"] == 0
        assert first["alternate"]["index"] == first["original"]["index"]

    def test_alternate_dataset_propagates_original_index(self):
        dataset = datasets.Dataset.from_dict(
            {
                "question": ["q0"],
                "answer": ["a0"],
                "alt_answer": ["alt-a0"],
            }
        )
        with patch("data.qa.load_hf_dataset", return_value=dataset), patch(
            "data.qa.preprocess_chat_instance", _fake_preprocess_chat_instance
        ):
            qa_dataset = QAwithAlternateDataset(
                hf_args={"path": "unused"},
                template_args=_make_template_args(),
                tokenizer=None,
                alternate_key="alt_answer",
            )
            item = qa_dataset[0]

        assert item["alternate"]["index"] == 0
        assert item["alternate"]["index"] == item["original"]["index"]

    def test_compute_unlearning_forget_losses_supports_npo_and_idkdpo(self):
        model = _FakeModel(_make_logits([(0, 1), (1, 0)]))
        ref_model = _FakeModel(_make_logits([(1, 1), (0, 0)]))

        npo_batch = {
            "input_ids": torch.tensor([[1, 2, 3], [1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            "labels": torch.tensor([[-100, 0, 1], [-100, 1, 0]]),
            "index": torch.tensor([10, 11]),
        }
        npo_indices, npo_losses = compute_unlearning_forget_losses(
            model=model, ref_model=ref_model, batch=npo_batch, method="NPO", beta=0.1
        )

        idkdpo_batch = {
            "original": npo_batch,
            "alternate": {
                "input_ids": torch.tensor([[1, 2, 3], [1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
                "labels": torch.tensor([[-100, 1, 1], [-100, 0, 0]]),
                "index": torch.tensor([10, 11]),
            },
        }
        idk_indices, idk_losses = compute_unlearning_forget_losses(
            model=model,
            ref_model=ref_model,
            batch=idkdpo_batch,
            method="IdkDPO",
            beta=0.1,
        )

        assert npo_indices == [10, 11]
        assert idk_indices == [10, 11]
        assert len(npo_losses) == 2
        assert len(idk_losses) == 2
        assert all(isinstance(loss, float) for loss in npo_losses + idk_losses)

    def test_compute_unlearning_forget_losses_reports_method_batch_mismatch(self):
        model = _FakeModel(_make_logits([(0, 1), (1, 0)]))
        ref_model = _FakeModel(_make_logits([(1, 1), (0, 0)]))
        idkdpo_batch = {
            "original": {
                "input_ids": torch.tensor([[1, 2, 3], [1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
                "labels": torch.tensor([[-100, 0, 1], [-100, 1, 0]]),
                "index": torch.tensor([10, 11]),
            },
            "alternate": {
                "input_ids": torch.tensor([[1, 2, 3], [1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
                "labels": torch.tensor([[-100, 1, 1], [-100, 0, 0]]),
                "index": torch.tensor([10, 11]),
            },
        }

        with pytest.raises(ValueError, match="method='npo'"):
            compute_unlearning_forget_losses(
                model=model,
                ref_model=ref_model,
                batch=idkdpo_batch,
                method="NPO",
                beta=0.1,
            )

    def test_selective_configs_allow_experiment_method_override(self):
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
            selective_cfg = compose(
                config_name="selective.yaml",
                overrides=["experiment=selective/tofu/idkdpo"],
            )

        assert selective_cfg.method == "IdkDPO"

    def test_difficulty_payload_and_stage_manifests_cover_missing_scores(self):
        payload = build_difficulty_payload(
            score_records={
                0: [0.1, 0.2],
                2: [0.4],
                1: [0.8],
            },
            metadata={
                "method": "NPO",
                "forget_split": "forget10",
                "model": "target-model",
                "all_indices": [0, 1, 2, 3],
            },
        )
        manifests = build_stage_manifests(
            difficulty_payload=payload,
            stage_percentiles=[0.5, 0.75, 1.0],
            stage_epoch_ratios=[0.2, 0.3, 0.5],
        )

        scores = payload["scores_by_index"]
        assert scores["0"]["num_refs"] == 2
        assert scores["0"]["rank"] < scores["2"]["rank"]
        assert scores["2"]["rank"] < scores["1"]["rank"]
        assert manifests[0]["allowed_indices"] == [0, 2]
        assert set(manifests[0]["allowed_indices"]).issubset(
            manifests[1]["allowed_indices"]
        )
        assert set(manifests[1]["allowed_indices"]).issubset(
            manifests[2]["allowed_indices"]
        )
        assert manifests[2]["allowed_indices"] == [0, 2, 1, 3]

    def test_apply_intra_stage_ordering_sets_dataset_and_sampler(self):
        cfg = OmegaConf.create(
            {
                "intra_stage_order": "strict",
                "data": {
                    "anchor": "forget",
                    "forget": {
                        "TOFU_QA_forget_idk": {
                            "args": {
                                "allowed_indices_path": "stage1.json",
                                "preserve_manifest_order": False,
                            }
                        }
                    },
                },
                "trainer": {
                    "train_sampler": "random",
                    "args": {"group_by_length": False},
                },
            }
        )

        with patch.dict("os.environ", {"WORLD_SIZE": "1"}, clear=False), patch(
            "selective.runtime.torch.cuda.device_count", return_value=1
        ):
            apply_intra_stage_ordering(cfg)

        assert cfg.trainer.train_sampler == "sequential"
        assert cfg.data.forget.TOFU_QA_forget_idk.args.preserve_manifest_order

    def test_apply_intra_stage_ordering_rejects_non_forget_anchor(self):
        cfg = OmegaConf.create(
            {
                "intra_stage_order": "strict",
                "data": {
                    "anchor": "retain",
                    "forget": {"TOFU_QA_forget_idk": {"args": {}}},
                },
                "trainer": {
                    "train_sampler": "random",
                    "args": {"group_by_length": False},
                },
            }
        )

        with patch("selective.runtime.torch.cuda.device_count", return_value=1):
            with pytest.raises(ValueError, match="data.anchor=forget"):
                apply_intra_stage_ordering(cfg)

    def test_apply_intra_stage_ordering_rejects_group_by_length(self):
        cfg = OmegaConf.create(
            {
                "intra_stage_order": "strict",
                "data": {
                    "anchor": "forget",
                    "forget": {"TOFU_QA_forget_idk": {"args": {}}},
                },
                "trainer": {
                    "train_sampler": "random",
                    "args": {"group_by_length": True},
                },
            }
        )

        with patch("selective.runtime.torch.cuda.device_count", return_value=1):
            with pytest.raises(ValueError, match="group_by_length"):
                apply_intra_stage_ordering(cfg)

    def test_apply_intra_stage_ordering_rejects_multi_process(self):
        cfg = OmegaConf.create(
            {
                "intra_stage_order": "strict",
                "data": {
                    "anchor": "forget",
                    "forget": {"TOFU_QA_forget_idk": {"args": {}}},
                },
                "trainer": {
                    "train_sampler": "random",
                    "args": {"group_by_length": False},
                },
            }
        )

        with patch.dict("os.environ", {"WORLD_SIZE": "2"}, clear=False), patch(
            "selective.runtime.torch.cuda.device_count", return_value=1
        ):
            with pytest.raises(ValueError, match="single-process"):
                apply_intra_stage_ordering(cfg)

    def test_finetune_trainer_uses_configured_train_sampler(self):
        trainer = object.__new__(FinetuneTrainer)
        trainer.train_dataset = [0, 1, 2]
        trainer.args = SimpleNamespace(group_by_length=False)
        trainer.processing_class = None

        trainer.train_sampler = "sequential"
        assert isinstance(trainer._get_train_sampler(), SequentialSampler)

        trainer.train_sampler = "random"
        assert isinstance(trainer._get_train_sampler(), RandomSampler)

    def test_prepare_difficulty_payload_reads_reference_manifest(self):
        dataset = datasets.Dataset.from_dict(
            {
                "question": ["q0", "q1", "q2", "q3"],
                "answer": ["a0", "a1", "a2", "a3"],
            }
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            models_dir = tmp_path / "models"
            for split_id in range(2):
                (models_dir / f"split{split_id}").mkdir(parents=True, exist_ok=True)

            split_manifests = build_reference_split_manifests(
                all_indices=[0, 1, 2, 3],
                output_dir=tmp_path / "folds",
                num_repeats=1,
                repeat_split_seed=0,
            )
            reference_manifest_path = tmp_path / "reference_models.json"
            build_reference_manifest(
                metadata={
                    "method": "NPO",
                    "model": "base-model",
                    "forget_split": "forget10",
                    "retain_split": "retain90",
                    "num_repeats": 1,
                    "repeat_split_seed": 0,
                    "all_indices": [0, 1, 2, 3],
                },
                reference_split_manifests=split_manifests,
                checkpoint_root_dir=models_dir,
                reference_manifest_output_path=reference_manifest_path,
                validate_checkpoint_paths=True,
            )

            with patch("data.qa.load_hf_dataset", return_value=dataset), patch(
                "data.qa.preprocess_chat_instance", _fake_preprocess_chat_instance
            ):
                qa_dataset = QADataset(
                    hf_args={"path": "unused"},
                    template_args=_make_template_args(),
                    tokenizer=None,
                )

            cfg = OmegaConf.create(
                {
                    "method": "NPO",
                    "beta": 0.1,
                    "batch_size": 4,
                    "forget_split": "forget10",
                    "reference_manifest_path": str(reference_manifest_path),
                    "model": {
                        "model_args": {
                            "pretrained_model_name_or_path": "base-model",
                        }
                    },
                }
            )

            seen_subsets = []

            def _score_dataset_with_reference(
                model, ref_model, dataset, collator, method, beta, batch_size
            ):
                subset_indices = list(dataset.data["index"])
                seen_subsets.append(subset_indices)
                return {int(idx): [float(idx)] for idx in subset_indices}

            with patch(
                "selective.pipeline._load_model_from_path",
                side_effect=lambda model_cfg, model_path: model_path,
            ), patch(
                "selective.pipeline.score_dataset_with_reference",
                side_effect=_score_dataset_with_reference,
            ):
                difficulty_payload = prepare_difficulty_payload(
                    cfg=cfg,
                    dataset=qa_dataset,
                    collator=object(),
                    target_model="base-model",
                )

            expected_subsets = [
                split_manifests[0]["heldout_indices"],
                split_manifests[1]["heldout_indices"],
            ]
            assert seen_subsets == expected_subsets
            assert difficulty_payload["metadata"]["reference_manifest_path"] == str(
                reference_manifest_path
            )
            assert difficulty_payload["metadata"]["num_reference_models"] == 2
            assert difficulty_payload["metadata"]["num_unscored_examples"] == 0
            assert difficulty_payload["metadata"]["coverage_fraction"] == 1.0
            assert sorted(
                int(idx) for idx in difficulty_payload["scores_by_index"].keys()
            ) == [0, 1, 2, 3]

    def test_selective_scripts_are_present_and_compilable(self):
        dpo_script = ROOT / "community" / "methods" / "Selective-DPO" / "run.py"
        npo_script = ROOT / "community" / "methods" / "Selective-NPO" / "run.py"

        assert dpo_script.is_file()
        assert npo_script.is_file()
        subprocess.run([sys.executable, "-m", "py_compile", str(dpo_script)], check=True)
        subprocess.run([sys.executable, "-m", "py_compile", str(npo_script)], check=True)

    def test_selective_scripts_add_non_schema_trainer_args_with_hydra_plus_prefix(self):
        dpo_script = (
            ROOT / "community" / "methods" / "Selective-DPO" / "run.py"
        ).read_text(encoding="utf-8")
        npo_script = (
            ROOT / "community" / "methods" / "Selective-NPO" / "run.py"
        ).read_text(encoding="utf-8")

        assert "+trainer.args.save_total_limit=1" in dpo_script
        assert "+trainer.args.save_total_limit=1" in npo_script
        assert "+trainer.args.ignore_data_skip=true" in dpo_script
        assert "+trainer.args.ignore_data_skip=true" in npo_script
        assert '"NUM_REFERENCE_REPEATS": "3"' in dpo_script
        assert '"NUM_REFERENCE_REPEATS": "3"' in npo_script
        assert '"INTRA_STAGE_ORDER": "random"' in dpo_script
        assert '"INTRA_STAGE_ORDER": "random"' in npo_script

    def test_selective_scripts_warm_start_next_stage_from_previous_model(self):
        dpo_script = (
            ROOT / "community" / "methods" / "Selective-DPO" / "run.py"
        ).read_text(encoding="utf-8")
        npo_script = (
            ROOT / "community" / "methods" / "Selective-NPO" / "run.py"
        ).read_text(encoding="utf-8")

        assert "stage_model_path = cfg.base_model_path" in dpo_script
        assert "stage_model_path = cfg.base_model_path" in npo_script
        assert "stage_model_path = str(latest_checkpoint)" in dpo_script
        assert "stage_model_path = str(latest_checkpoint)" in npo_script
        assert (
            'f"model.model_args.pretrained_model_name_or_path={pretrained_model_path}"'
            in dpo_script
        )
        assert (
            'f"model.model_args.pretrained_model_name_or_path={pretrained_model_path}"'
            in npo_script
        )

    def test_selective_scripts_use_short_output_prefixes(self):
        dpo_script = (
            ROOT / "community" / "methods" / "Selective-DPO" / "run.py"
        ).read_text(encoding="utf-8")
        npo_script = (
            ROOT / "community" / "methods" / "Selective-NPO" / "run.py"
        ).read_text(encoding="utf-8")

        assert (
            'return f"tofu_{self.model}_{self.forget_split}_{self.reference_prepare_config_suffix}"'
            in dpo_script
        )
        assert (
            'return f"{self.reference_task_prefix}_references"'
            in npo_script
        )
        assert 'return self.reference_dir / "reference_splits"' in dpo_script
        assert 'return self.reference_dir / "reference_splits"' in npo_script
        assert 'f"{self.stage_task_base_prefix}_{self.intra_stage_order}"' in dpo_script
        assert 'f"{self.stage_task_base_prefix}_{self.intra_stage_order}"' in npo_script
