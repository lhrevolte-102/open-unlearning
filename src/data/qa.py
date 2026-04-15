import torch
from torch.utils.data import Dataset

from data.utils import (
    add_dataset_index,
    filter_dataset_by_index,
    load_allowed_indices,
    load_hf_dataset,
    preprocess_chat_instance,
)


class QADataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        question_key="question",
        answer_key="answer",
        few_shot_dataset_hf_args=None,
        max_length=512,
        predict_with_generate=False,
        allowed_indices_path=None,
    ):
        super(QADataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        if allowed_indices_path is not None:
            allowed_indices = load_allowed_indices(allowed_indices_path)
            self.data = filter_dataset_by_index(self.data, allowed_indices)
        self.fs_data = None
        if few_shot_dataset_hf_args is not None:
            raw_data = load_hf_dataset(**few_shot_dataset_hf_args)
            self.fs_data = {}
            self.fs_data[question_key] = raw_data[question_key]
            self.fs_data[answer_key] = raw_data[answer_key]
        self.template_args = template_args
        self.question_key = question_key
        self.answer_key = answer_key
        self.predict_with_generate = predict_with_generate

    def __len__(self):
        return len(self.data)

    def _process_sample(self, question, answer, index=-1):
        if self.fs_data is None:
            prompt_msgs, response_msgs = [question], [answer]
        else:
            prompt_msgs = self.fs_data[self.question_key] + [question]
            response_msgs = self.fs_data[self.answer_key] + [answer]
        tokenized_data = preprocess_chat_instance(
            self.tokenizer,
            self.template_args,
            prompt_msgs,
            response_msgs,
            self.max_length,
            self.predict_with_generate,
        )
        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
            "index": index,
        }
        return item_dct

    def __getitem__(self, idx):
        question = self.data[idx][self.question_key]
        answer = self.data[idx][self.answer_key]
        index = self.data[idx]["index"]
        if isinstance(answer, str):
            item = self._process_sample(question=question, answer=answer, index=index)
        elif isinstance(answer, list):
            item = {}
            for i, ans in enumerate(answer):
                sample_item = self._process_sample(
                    question=question, answer=ans, index=index
                )
                item[i] = sample_item
        else:
            raise NotImplementedError("answer format not found")
        return item


class QAwithIdkDataset(QADataset):
    def __init__(
        self,
        idk_path,
        return_original=True,
        idk_sampling_mode="random",
        idk_sampling_seed=0,
        *args,
        **kwargs,
    ):
        self.idk_path = idk_path
        self.return_original = return_original
        self.idk_sampling_mode = idk_sampling_mode
        self.idk_sampling_seed = idk_sampling_seed
        with open(self.idk_path, "r", encoding="utf-8") as file:
            self.idk_responses = file.readlines()
        super().__init__(*args, **kwargs)

    def item_with_idk(self, question, index=-1):
        if self.idk_sampling_mode == "random":
            rand_pos = torch.randint(0, len(self.idk_responses), (1,)).item()
        elif self.idk_sampling_mode == "deterministic":
            rand_pos = (int(index) + int(self.idk_sampling_seed)) % len(
                self.idk_responses
            )
        else:
            raise ValueError(
                f"Unsupported idk_sampling_mode '{self.idk_sampling_mode}', expected "
                "'random' or 'deterministic'"
            )
        idk_response = self.idk_responses[rand_pos].strip()
        idk_item = self._process_sample(
            question=question, answer=idk_response, index=index
        )
        return idk_item

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        question = self.data[idx][self.question_key]
        index = self.data[idx]["index"]
        if isinstance(item, dict):
            return_item = {"original": item}
            idk_item = self.item_with_idk(question, index=index)
            return_item["alternate"] = idk_item
            # return_item = [item, idk_item]
        elif isinstance(item, list) or isinstance(item, tuple):
            return_item = []
            for sample_item in item:
                return_item = {"original": sample_item}
                idk_item = self.item_with_idk(question, index=index)
                return_item["alternate"] = idk_item
                # return_item.append([sample_item, idk_item])
        return return_item if self.return_original else return_item["alternate"]


class QAwithAlternateDataset(QADataset):
    def __init__(self, alternate_key, return_original=True, *args, **kwargs):
        self.alternate_key = alternate_key
        self.return_original = return_original
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        question = self.data[idx][self.question_key]
        index = self.data[idx]["index"]
        if isinstance(item, dict):
            return_item = {"original": item}
            alt_item = self._process_sample(
                question=question,
                answer=self.data[idx][self.alternate_key],
                index=index,
            )
            return_item["alternate"] = alt_item
            # return_item = [item, idk_item]
        elif isinstance(item, list) or isinstance(item, tuple):
            return_item = []
            for sample_item in item:
                return_item = {"original": sample_item}
                alt_item = self._process_sample(
                    question=question,
                    answer=self.data[idx][self.alternate_key],
                    index=index,
                )
                return_item["alternate"] = alt_item
                # return_item.append([sample_item, idk_item])
        return return_item if self.return_original else return_item["alternate"]
