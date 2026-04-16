import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import datasets
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import RandomSampler, SequentialSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.qa import QADataset, QAwithAlternateDataset, QAwithIdkDataset
from data.utils import load_allowed_indices
from selective.runtime import apply_intra_stage_ordering
from selective.utils import (
    build_fold_manifests,
    build_reference_manifest,
    build_stage_manifests,
    load_json,
)
from selective.score import build_difficulty_payload, compute_unlearning_forget_losses
from selective_prepare import prepare_difficulty_payload
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


class SelectiveTests(unittest.TestCase):
    def test_reference_fold_manifests_partition_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fold_manifests = build_fold_manifests(
                all_indices=[0, 1, 2, 3, 4, 5],
                num_folds=3,
                fold_assignment_seed=0,
                output_dir=tmp_dir,
            )

            heldout_union = set()
            for fold_manifest in fold_manifests:
                train_payload = load_json(fold_manifest["train_manifest_path"])
                heldout_payload = load_json(fold_manifest["heldout_manifest_path"])
                train_indices = set(train_payload["allowed_indices"])
                heldout_indices = set(heldout_payload["allowed_indices"])

                self.assertFalse(train_indices & heldout_indices)
                self.assertEqual(train_indices | heldout_indices, set(range(6)))
                heldout_union |= heldout_indices

            self.assertEqual(heldout_union, set(range(6)))

    def test_reference_manifest_contains_expected_records(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            models_dir = tmp_path / "models"
            for fold_id in range(2):
                (models_dir / f"fold{fold_id}").mkdir(parents=True, exist_ok=True)

            fold_manifests = build_fold_manifests(
                all_indices=[0, 1, 2, 3],
                num_folds=2,
                fold_assignment_seed=0,
                output_dir=tmp_path / "folds",
            )
            reference_manifest = build_reference_manifest(
                metadata={
                    "method": "NPO",
                    "model": "base-model",
                    "forget_split": "forget10",
                    "retain_split": "retain90",
                    "num_folds": 2,
                    "all_indices": [0, 1, 2, 3],
                },
                fold_manifests=fold_manifests,
                checkpoint_root_dir=models_dir,
                reference_manifest_output_path=tmp_path / "reference_models.json",
                validate_checkpoint_paths=True,
            )

            self.assertEqual(reference_manifest["metadata"]["num_reference_models"], 2)
            self.assertEqual(len(reference_manifest["references"]), 2)
            self.assertTrue(
                reference_manifest["references"][0]["checkpoint_path"].endswith("fold0")
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

        self.assertEqual(len(qa_dataset), 2)
        self.assertEqual(qa_dataset.data["index"], [0, 2])

    def test_load_allowed_indices_can_preserve_manifest_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "stage1.json"
            manifest_path.write_text(
                json.dumps({"allowed_indices": [2, 0, 2, 1]}), encoding="utf-8"
            )

            self.assertEqual(load_allowed_indices(manifest_path), [0, 1, 2])
            self.assertEqual(
                load_allowed_indices(manifest_path, preserve_order=True), [2, 0, 1]
            )

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

        self.assertEqual(len(qa_dataset), 3)
        self.assertEqual(qa_dataset.data["index"], [2, 0, 1])

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

        self.assertTrue(
            torch.equal(first["alternate"]["input_ids"], second["alternate"]["input_ids"])
        )
        self.assertEqual(first["alternate"]["index"], 0)
        self.assertEqual(first["alternate"]["index"], first["original"]["index"])

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

        self.assertEqual(item["alternate"]["index"], 0)
        self.assertEqual(item["alternate"]["index"], item["original"]["index"])

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

        self.assertEqual(npo_indices, [10, 11])
        self.assertEqual(idk_indices, [10, 11])
        self.assertEqual(len(npo_losses), 2)
        self.assertEqual(len(idk_losses), 2)
        self.assertTrue(all(isinstance(loss, float) for loss in npo_losses + idk_losses))

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

        with self.assertRaisesRegex(ValueError, "method='npo'"):
            compute_unlearning_forget_losses(
                model=model,
                ref_model=ref_model,
                batch=idkdpo_batch,
                method="NPO",
                beta=0.1,
            )

    def test_selective_configs_allow_experiment_method_override(self):
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
            prepare_cfg = compose(
                config_name="selective_prepare.yaml",
                overrides=["experiment=selective/tofu/idkdpo"],
            )
            reference_cfg = compose(
                config_name="selective_reference.yaml",
                overrides=["experiment=selective/tofu/idkdpo"],
            )

        self.assertEqual(prepare_cfg.method, "IdkDPO")
        self.assertEqual(reference_cfg.method, "IdkDPO")

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
        self.assertLess(scores["0"]["rank"], scores["2"]["rank"])
        self.assertLess(scores["2"]["rank"], scores["1"]["rank"])
        self.assertEqual(manifests[0]["allowed_indices"], [0, 2])
        self.assertTrue(
            set(manifests[0]["allowed_indices"]).issubset(manifests[1]["allowed_indices"])
        )
        self.assertTrue(
            set(manifests[1]["allowed_indices"]).issubset(manifests[2]["allowed_indices"])
        )
        self.assertEqual(manifests[2]["allowed_indices"], [0, 2, 1, 3])

    def test_apply_intra_stage_ordering_sets_dataset_and_sampler(self):
        cfg = OmegaConf.create(
            {
                "intra_stage_order": "difficulty_strict",
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

        self.assertEqual(cfg.trainer.train_sampler, "sequential")
        self.assertTrue(
            cfg.data.forget.TOFU_QA_forget_idk.args.preserve_manifest_order
        )

    def test_apply_intra_stage_ordering_rejects_non_forget_anchor(self):
        cfg = OmegaConf.create(
            {
                "intra_stage_order": "difficulty_strict",
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
            with self.assertRaisesRegex(ValueError, "data.anchor=forget"):
                apply_intra_stage_ordering(cfg)

    def test_apply_intra_stage_ordering_rejects_group_by_length(self):
        cfg = OmegaConf.create(
            {
                "intra_stage_order": "difficulty_strict",
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
            with self.assertRaisesRegex(ValueError, "group_by_length"):
                apply_intra_stage_ordering(cfg)

    def test_apply_intra_stage_ordering_rejects_multi_process(self):
        cfg = OmegaConf.create(
            {
                "intra_stage_order": "difficulty_strict",
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
            with self.assertRaisesRegex(ValueError, "single-process"):
                apply_intra_stage_ordering(cfg)

    def test_finetune_trainer_uses_configured_train_sampler(self):
        trainer = object.__new__(FinetuneTrainer)
        trainer.train_dataset = [0, 1, 2]
        trainer.args = SimpleNamespace(group_by_length=False)
        trainer.processing_class = None

        trainer.train_sampler = "sequential"
        self.assertIsInstance(trainer._get_train_sampler(), SequentialSampler)

        trainer.train_sampler = "random"
        self.assertIsInstance(trainer._get_train_sampler(), RandomSampler)

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
            for fold_id in range(2):
                (models_dir / f"fold{fold_id}").mkdir(parents=True, exist_ok=True)

            fold_manifests = build_fold_manifests(
                all_indices=[0, 1, 2, 3],
                num_folds=2,
                fold_assignment_seed=0,
                output_dir=tmp_path / "folds",
            )
            reference_manifest_path = tmp_path / "reference_models.json"
            build_reference_manifest(
                metadata={
                    "method": "NPO",
                    "model": "base-model",
                    "forget_split": "forget10",
                    "retain_split": "retain90",
                    "num_folds": 2,
                    "all_indices": [0, 1, 2, 3],
                },
                fold_manifests=fold_manifests,
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
                "selective_prepare._load_model_from_path",
                side_effect=lambda model_cfg, model_path: model_path,
            ), patch(
                "selective_prepare.score_dataset_with_reference",
                side_effect=_score_dataset_with_reference,
            ):
                difficulty_payload = prepare_difficulty_payload(
                    cfg=cfg,
                    dataset=qa_dataset,
                    collator=object(),
                    target_model="base-model",
                )

            expected_subsets = [
                fold_manifests[0]["heldout_indices"],
                fold_manifests[1]["heldout_indices"],
            ]
            self.assertEqual(seen_subsets, expected_subsets)
            self.assertEqual(
                difficulty_payload["metadata"]["reference_manifest_path"],
                str(reference_manifest_path),
            )
            self.assertEqual(difficulty_payload["metadata"]["num_reference_models"], 2)
            self.assertEqual(
                sorted(int(idx) for idx in difficulty_payload["scores_by_index"].keys()),
                [0, 1, 2, 3],
            )

    def test_selective_scripts_have_valid_bash_syntax(self):
        subprocess.run(
            ["bash", "-n", str(ROOT / "community" / "methods" / "Selective-DPO" / "run.sh")],
            check=True,
        )
        subprocess.run(
            ["bash", "-n", str(ROOT / "community" / "methods" / "Selective-NPO" / "run.sh")],
            check=True,
        )

    def test_selective_scripts_add_non_schema_trainer_args_with_hydra_plus_prefix(self):
        dpo_script = (
            ROOT / "community" / "methods" / "Selective-DPO" / "run.sh"
        ).read_text(encoding="utf-8")
        npo_script = (
            ROOT / "community" / "methods" / "Selective-NPO" / "run.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("+trainer.args.save_total_limit=1", dpo_script)
        self.assertIn("+trainer.args.save_total_limit=1", npo_script)
        self.assertIn("+trainer.args.ignore_data_skip=true", dpo_script)
        self.assertIn("+trainer.args.ignore_data_skip=true", npo_script)


if __name__ == "__main__":
    unittest.main()
