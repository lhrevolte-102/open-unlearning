import subprocess
import socket
import os


def get_free_port():
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


model = "Llama-3.2-3B-Instruct"
forget_split, holdout_split, retain_split = "forget10", "holdout10", "retain90"

# Unlearn only needs 2 unique tasks: base / pr
unlearn_conditions = [
    ("base", "baseline"),
    ("pr", "prompt_repetition"),
]

# Eval keeps 2 conditions; combined with unlearn -> 4 cases
eval_conditions = [
    ("base", "baseline"),
    ("pr", "prompt_repetition"),
]

trainers_args = [
    ("DPO", {"alpha": 1, "beta": 0.1})
    # ("NPO", {"alpha": 1, "beta": 0.05}),
    # ("NPO", {"alpha": 1, "beta": 0.1}),
    # ("NPO", {"alpha": 1, "beta": 0.5}),
    # ("NPO", {"alpha": 5, "beta": 0.1}),
    # ("NPO", {"alpha": 5, "beta": 0.5}),
    # ("RMU", {"alpha": 1, "steering_coeff": 1}),
    # ("RMU", {"alpha": 1, "steering_coeff": 10}),
    # ("RMU", {"alpha": 10, "steering_coeff": 1}),
    # ("RMU", {"alpha": 10, "steering_coeff": 10}),
]

master_port = get_free_port()
print(f"Master Port: {master_port}")

for trainer, args in trainers_args:
    args_suffix = "_".join(f"{k}{v}" for k, v in args.items())
    method_args_overrides = [f"trainer.method_args.{k}={v}" for k, v in args.items()]
    experiment_cfg = "unlearn/tofu/idk.yaml" if trainer == "DPO" else "unlearn/tofu/default.yaml"

    for unlearn_name, unlearn_condition in unlearn_conditions:
        unlearn_task_name = (
            f"tofu_{model}_{forget_split}_{trainer}_{args_suffix}_{unlearn_name}"
        )
        unlearn_save_path = f"saves/unlearn/{unlearn_task_name}"
        unlearn_condition_overrides = [
            f"+model.template_args.condition={unlearn_condition}"
        ]

        # Unlearn command (run once per base/pr)
        if os.path.exists(unlearn_save_path):
            print(
                f"\n[Unlearn] Skipping {unlearn_task_name} (already exists at {unlearn_save_path})"
            )
        else:
            unlearn_cmd = (
                [
                    "env",
                    "CUDA_VISIBLE_DEVICES=0,1",
                    "accelerate",
                    "launch",
                    "--config_file",
                    "configs/accelerate/default_config.yaml",
                    "--main_process_port",
                    str(master_port),
                    "src/train.py",
                    "--config-name=unlearn.yaml",
                    f"task_name={unlearn_task_name}",
                    f"experiment={experiment_cfg}",
                    f"forget_split={forget_split}",
                    f"retain_split={retain_split}",
                    f"model={model}",
                    f"model.model_args.pretrained_model_name_or_path=../models/tofu_{model}_full",
                    f"model.tokenizer_args.pretrained_model_name_or_path=../models/tofu_{model}_full",
                    f"trainer={trainer}",
                    "trainer.args.eval_on_start=false",
                    "trainer.args.do_eval=false",
                    "trainer.args.eval_strategy=no",
                    "trainer.args.ddp_find_unused_parameters=true",
                    "trainer.args.gradient_checkpointing=true",
                    "trainer.args.per_device_train_batch_size=8",
                    "trainer.args.gradient_accumulation_steps=4",
                    "trainer.args.num_train_epochs=10",
                ]
                + method_args_overrides
                + unlearn_condition_overrides
            )

            print(f"\n[Unlearn] {unlearn_task_name}")
            subprocess.run(
                " ".join(unlearn_cmd),
                shell=True,
                check=True,
                env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "0,1"},
            )

        # Eval command (4 combinations total)
        for eval_name, eval_condition in eval_conditions:
            eval_task_name = f"tofu_{model}_{forget_split}_{trainer}_{args_suffix}_{unlearn_name}_{eval_name}"
            eval_condition_overrides = [
                f"+model.template_args.condition={eval_condition}"
            ]

            eval_cmd = [
                "python",
                "src/eval.py",
                "experiment=eval/tofu/default.yaml",
                f"forget_split={forget_split}",
                f"holdout_split={holdout_split}",
                f"model={model}",
                f"task_name={eval_task_name}",
                f"model.model_args.pretrained_model_name_or_path={unlearn_save_path}",
                f"model.tokenizer_args.pretrained_model_name_or_path={unlearn_save_path}",
                f"retain_logs_path=saves/eval/tofu_{model}_{retain_split}/TOFU_EVAL.json",
                "eval.tofu.metrics.forget_Q_A_ROUGE.generation_args.do_sample=true",
                "eval.tofu.metrics.forget_Q_A_ROUGE.generation_args.temperature=2.0",
                "+eval.tofu.metrics.forget_Q_A_ROUGE.generation_args.top_k=40",
                "eval.tofu.metrics.forget_Q_A_PARA_ROUGE.generation_args.do_sample=true",
                "eval.tofu.metrics.forget_Q_A_PARA_ROUGE.generation_args.temperature=2.0",
                "+eval.tofu.metrics.forget_Q_A_PARA_ROUGE.generation_args.top_k=40",
            ] + eval_condition_overrides

            print(f"[Eval] {eval_task_name}")
            subprocess.run(
                " ".join(eval_cmd),
                shell=True,
                check=True,
                env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "0"},
            )
