import torch
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='RL')

    # dataset
    parser.add_argument(
        '--output-dir', type=str, default='outputs')
    parser.add_argument(
        '--dataset-train', type=str, default='data/CLOTH/train.json',
        help='JSONL file containing train prompts. Each row must contain a prompt at `row["prompt"]["text"]`.')
    parser.add_argument(
        '--dataset-val', type=str, default='data/CLOTH/valid.json',
        help='JSONL file containing dev prompts. Each row must contain a prompt at `row["prompt"]["text"]`.')
    parser.add_argument(
        '--perspective-rate-limit', type=int, default=135, help='number of perspective call per second')

    # reward
    parser.add_argument(
        '--reward_model_path', type=str, default='model/reward/model.pt', help='reward model')
    parser.add_argument(
        '--n_extra_tokens', type=int, default=5, help='number of reward categorization')
    parser.add_argument(
        '--sample_interval', type=int, default=500, help='step interval to sample from current policy')
    parser.add_argument(
        '--horizon', type=float, default=2500, help='horizon value in adaptive controller')
    # KL term
    parser.add_argument(
        '--kl_coef', type=float, default=0.05, help='coefficient for KL term in reward')
    parser.add_argument(
        '--adaptive_kl', action='store_true', default=False, help='whether to use adaptive KL controller')
    parser.add_argument(
        '--target_kl', type=float, default=3, help='target value in adaptive KL controller')
    # entropy term
    parser.add_argument(
        '--entropy_coef', type=float, default=0.06, help='coefficient for entropy term in reward')
    parser.add_argument(
        '--adaptive_entropy', action='store_true', default=False, help='whether to use adaptive entropy controller')
    parser.add_argument(
        '--target_entropy', type=float, default=40, help='target value in adaptive entropy controller')

    # policy
    parser.add_argument(
        '--init-model', type=str, default='model/policy', help='language model used for policy.')
    parser.add_argument(
        '--ref-model', type=str, default='model/policy', help='language model used for reference policy.')
    parser.add_argument(
        '--response-length', type=int, default=20, help='number of tokens to generate for each prompt.')
    parser.add_argument(
        '--temperature', type=float, default=1.0, help='temperature for sampling policy.')

    # training˚
    parser.add_argument(
        '--total_episodes', type=int, default=3000000, help='total number of episodes')
    parser.add_argument(
        '--batch_size', type=int, default=128, help='batch size')
    parser.add_argument(
        '--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument(
        '--num_warmup_steps', type=int, default=500, help='number of warmup steps in lr scheduler')
    parser.add_argument(
        '--clip_grad', action='store_true', default=False, help='whether to clip gradient')
    parser.add_argument(
        '--max-grad-norm', type=float, default=0.5, help='maximum norm of gradients ')
    parser.add_argument(
        '--accimulation_steps', type=int, default=1, help='gradient accumulation')

    # generation
    parser.add_argument(
        '--sample_num', type=int, default=5, help='number of samples to generate for each prompt.')
    parser.add_argument(
        '--top-p', type=float, default=1.0, help='hyperparameter for nucleus sampling')

    # other
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--log_interval', type=int, default=100, help='step interval to print out logs')
    parser.add_argument(
        '--save_interval', type=int, default=1000, help='step interval to save model checkpoints')
    parser.add_argument(
        '--eval_interval', type=int, default=500, help='step interval to do evaluation')
    parser.add_argument(
        '--cuda_deterministic', action='store_false', default=True,
        help="sets flags for determinism when using CUDA (potentially slow!)")

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    return args
