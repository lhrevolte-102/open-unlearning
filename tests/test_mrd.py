import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.unlearn import ForgetRetainDataset
from mrd_utils import build_mrd_payload, score_dataset_with_mrd
from trainer.base import FinetuneTrainer


class _ToyDataset(torch.utils.data.Dataset):
    def __init__(self, indices, labels):
        self.indices = list(indices)
        self.labels = list(labels)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        label_values = self.labels[idx]
        return {
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
            "labels": torch.tensor(label_values, dtype=torch.long),
            "index": int(self.indices[idx]),
        }


def _toy_collator(samples):
    return {
        "input_ids": torch.stack([sample["input_ids"] for sample in samples]),
        "attention_mask": torch.stack([sample["attention_mask"] for sample in samples]),
        "labels": torch.stack([sample["labels"] for sample in samples]),
        "index": torch.tensor([sample["index"] for sample in samples], dtype=torch.long),
    }


class _ToyLM(torch.nn.Module):
    def __init__(self, vocab_size=8, hidden_size=4):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.proj = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        hidden = self.embed(input_ids)
        logits = self.proj(hidden)
        return SimpleNamespace(logits=logits)


class TestMRD:
    def test_mrd_scorer_is_stable_with_fixed_seed(self):
        torch.manual_seed(0)
        model = _ToyLM()
        dataset = _ToyDataset(
            indices=[5, 2],
            labels=[[-100, 1, 2], [-100, 2, 3]],
        )

        first_scores = score_dataset_with_mrd(
            model=model,
            dataset=dataset,
            collator=_toy_collator,
            batch_size=2,
            sigma=1e-3,
            num_mc_samples=4,
            eps=1e-6,
            seed=7,
        )
        second_scores = score_dataset_with_mrd(
            model=model,
            dataset=dataset,
            collator=_toy_collator,
            batch_size=2,
            sigma=1e-3,
            num_mc_samples=4,
            eps=1e-6,
            seed=7,
        )

        assert first_scores == second_scores
        payload = build_mrd_payload(
            scores_by_index=first_scores,
            metadata={"model_path": "toy-model", "forget_split": "forget10"},
        )
        assert sorted(payload["scores_by_index"].keys()) == ["2", "5"]
        assert sorted(payload["weights_by_index"].keys()) == ["2", "5"]
        assert "score" in payload["scores_by_index"]["2"]

    def test_weighted_sampler_aligns_weights_to_anchor_indices(self):
        forget_dataset = _ToyDataset(
            indices=[7, 3, 5],
            labels=[[-100, 1, 2], [-100, 2, 3], [-100, 3, 4]],
        )
        train_dataset = ForgetRetainDataset(
            forget=forget_dataset, retain=None, anchor="forget"
        )

        trainer = object.__new__(FinetuneTrainer)
        trainer.train_dataset = train_dataset
        trainer.args = SimpleNamespace(group_by_length=False)
        trainer.processing_class = None
        trainer.train_sampler = "weighted"

        with tempfile.TemporaryDirectory() as tmp_dir:
            weights_path = Path(tmp_dir) / "difficulty.json"
            weights_path.write_text(
                '{"weights_by_index":{"3":0.3,"5":0.5,"7":0.7}}',
                encoding="utf-8",
            )
            trainer.train_sampler_args = {
                "weights_path": str(weights_path),
                "replacement": True,
                "num_samples": 3,
            }
            sampler = trainer._get_train_sampler()

        assert sampler.weights.tolist() == [0.7, 0.3, 0.5]
        assert sampler.num_samples == 3

    def test_mrd_configs_compose(self):
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
            mrd_cfg = compose(
                config_name="mrd.yaml",
                overrides=["experiment=mrd/tofu/npo"],
            )
            unlearn_cfg = compose(
                config_name="unlearn.yaml",
                overrides=["experiment=unlearn/tofu/mrd_npo"],
            )

        assert mrd_cfg.forget_split == "forget10"
        assert unlearn_cfg.trainer.train_sampler == "weighted"
        assert "weights_path" in unlearn_cfg.trainer.train_sampler_args

    def test_mrd_run_script_is_present_and_compilable(self):
        script_path = ROOT / "community" / "methods" / "MRD-NPO" / "run.py"
        script_text = script_path.read_text(encoding="utf-8")

        assert "REFRESH_EPOCHS" in script_text
        assert "src/mrd.py" in script_text
        assert "trainer.train_sampler_args.weights_path" in script_text
        subprocess.run([sys.executable, "-m", "py_compile", str(script_path)], check=True)
