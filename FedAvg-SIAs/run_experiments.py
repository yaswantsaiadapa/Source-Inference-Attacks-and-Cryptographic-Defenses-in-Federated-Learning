import os
import copy
import argparse
import numpy as np
import pandas as pd
import torch

from models.Nets import Mnistcnn
from models.Update import LocalUpdate
from models.test import test_img
from models.Fed import FedAvg
from models.Sia import SIA
from models.defenses import apply_differential_privacy, apply_secure_aggregation
from utils.dataset import get_dataset
from utils.options import args_parser


def run_single_experiment(args, defense='none', dp_noise=0.0):
    """
    Executes a federated learning training loop with SIA attack evaluation
    under a specified defense mechanism (none, dp, secagg, dp+secagg).
    """
    # Set seed
    torch.manual_seed(args.manualseed)
    np.random.seed(args.manualseed)
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    dataset_train, dataset_test, dict_party_user, dict_sample_user = get_dataset(args)

    net_glob = Mnistcnn(args=args).to(args.device)
    empty_net = copy.deepcopy(net_glob)
    w_glob = net_glob.state_dict()

    size_per_client = [len(dict_party_user[i]) for i in range(args.num_users)]
    total_size = sum(size_per_client)
    size_weight = np.array(size_per_client) / total_size

    best_attack_acc = 0.0
    final_test_acc = 0.0

    for iter in range(args.epochs):
        w_locals = []
        loss_locals = []

        # Local client updates
        for idx in range(args.num_users):
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(loss)

        # Apply defense to transmitted weights
        if defense == 'none':
            w_transmitted = copy.deepcopy(w_locals)
            w_glob = FedAvg(w_locals, size_weight)

        elif defense == 'dp':
            # Local DP adds calibrated noise before transmission
            w_transmitted = apply_differential_privacy(w_locals, noise_multiplier=dp_noise)
            # Server aggregates noisy updates
            w_glob = FedAvg(w_transmitted, size_weight)

        elif defense == 'secagg':
            # Secure Aggregation blinds each client with pairwise zero-sum masks
            masked_w, w_secagg_avg = apply_secure_aggregation(w_locals, size_weight)
            # Attacker only observes masked models
            w_transmitted = masked_w
            # Server recovers exact unmasked sum via cancellation
            w_glob = w_secagg_avg

        elif defense == 'dp_secagg':
            # Defense-in-depth: DP noise added, then masked by SecAgg
            dp_w = apply_differential_privacy(w_locals, noise_multiplier=dp_noise)
            masked_w, w_secagg_avg = apply_secure_aggregation(dp_w, size_weight)
            w_transmitted = masked_w
            w_glob = w_secagg_avg
        else:
            raise ValueError(f"Unknown defense type: {defense}")

        # Attacker evaluates Source Inference Attack against transmitted weights
        s_attack = SIA(args=args, w_locals=w_transmitted, dataset=dataset_train, dict_mia_users=dict_sample_user)
        attack_acc = s_attack.attack(net=empty_net.to(args.device))
        best_attack_acc = max(best_attack_acc, attack_acc)

        # Update global model
        net_glob.load_state_dict(w_glob)

    # Evaluate final global model accuracy on unseen test data
    net_glob.eval()
    final_test_acc, _ = test_img(net_glob, dataset_test, args)

    return float(final_test_acc), float(best_attack_acc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2, help='Communication rounds')
    parser.add_argument('--local_ep', type=int, default=1, help='Local epochs per round')
    parser.add_argument('--num_users', type=int, default=10, help='Total clients (K)')
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--model', type=str, default='cnn')
    parser.add_argument('--results_csv', type=str, default='all_experiments_results.csv')
    cli_args = parser.parse_args()

    results_file = cli_args.results_csv

    # Initialize results dataframe or load existing
    if os.path.exists(results_file):
        df_results = pd.read_csv(results_file)
        print(f"Resuming experiments from {results_file} with {len(df_results)} existing records.")
    else:
        df_results = pd.DataFrame(columns=[
            'phase', 'defense', 'alpha', 'local_ep', 'seed', 'dp_noise',
            'test_acc', 'attack_acc', 'random_guess_acc'
        ])

    def already_done(defense, alpha, seed, dp_noise):
        matches = df_results[
            (df_results['defense'] == defense) &
            (np.isclose(df_results['alpha'].astype(float), alpha)) &
            (df_results['seed'].astype(int) == seed) &
            (np.isclose(df_results['dp_noise'].astype(float), dp_noise))
        ]
        return len(matches) > 0

    base_args = args_parser()
    base_args.epochs = cli_args.epochs
    base_args.local_ep = cli_args.local_ep
    base_args.num_users = cli_args.num_users
    base_args.dataset = cli_args.dataset
    base_args.model = cli_args.model
    base_args.gpu = -1  # force CPU
    base_args.all_clients = True

    experiments_queue = []

    # ============================================================
    # PHASE 2: Baseline Sweep (FedAvg + SIA)
    # Alphas: [0.1, 1.0, 10.0], Seeds: [1, 2]
    # ============================================================
    for alpha in [0.1, 1.0, 10.0]:
        for seed in [1, 2]:
            experiments_queue.append({
                'phase': 'Phase 2: Baseline',
                'defense': 'none',
                'alpha': alpha,
                'seed': seed,
                'dp_noise': 0.0
            })

    # ============================================================
    # PHASE 3: Differential Privacy Sweep
    # Primary config: alpha=0.1 (strongest SIA attack), Seeds: [1, 2]
    # DP noise multipliers: [0.001, 0.01, 0.05, 0.1]
    # ============================================================
    for dp_noise in [0.001, 0.01, 0.05, 0.1]:
        for seed in [1, 2]:
            experiments_queue.append({
                'phase': 'Phase 3: DP Defense',
                'defense': 'dp',
                'alpha': 0.1,
                'seed': seed,
                'dp_noise': dp_noise
            })

    # ============================================================
    # PHASE 4: Secure Aggregation
    # Primary config: alpha=0.1, Seeds: [1, 2]
    # ============================================================
    for seed in [1, 2]:
        experiments_queue.append({
            'phase': 'Phase 4: SecAgg Defense',
            'defense': 'secagg',
            'alpha': 0.1,
            'seed': seed,
            'dp_noise': 0.0
        })

    # Combined Defense: DP + SecAgg (Defense-in-depth)
    for seed in [1, 2]:
        experiments_queue.append({
            'phase': 'Phase 4: DP + SecAgg',
            'defense': 'dp_secagg',
            'alpha': 0.1,
            'seed': seed,
            'dp_noise': 0.01
        })

    total_runs = len(experiments_queue)
    print(f"\n=======================================================")
    print(f"Total Scheduled Runs: {total_runs}")
    print(f"Dataset: MNIST | Clients: {base_args.num_users} | Rounds: {base_args.epochs} | Local Ep: {base_args.local_ep}")
    print(f"Random Guessing Baseline: {100.0 / base_args.num_users:.1f}%")
    print(f"=======================================================\n")

    for i, exp in enumerate(experiments_queue, 1):
        if already_done(exp['defense'], exp['alpha'], exp['seed'], exp['dp_noise']):
            print(f"[{i}/{total_runs}] Skipping already completed: {exp['defense']} (alpha={exp['alpha']}, seed={exp['seed']}, dp_noise={exp['dp_noise']})")
            continue

        print(f"\n>>> [{i}/{total_runs}] Running {exp['phase']} | Defense={exp['defense']} | alpha={exp['alpha']} | seed={exp['seed']} | dp_noise={exp['dp_noise']}")
        
        run_args = copy.deepcopy(base_args)
        run_args.alpha = exp['alpha']
        run_args.manualseed = exp['seed']

        test_acc, attack_acc = run_single_experiment(
            args=run_args,
            defense=exp['defense'],
            dp_noise=exp['dp_noise']
        )

        new_row = {
            'phase': exp['phase'],
            'defense': exp['defense'],
            'alpha': exp['alpha'],
            'local_ep': base_args.local_ep,
            'seed': exp['seed'],
            'dp_noise': exp['dp_noise'],
            'test_acc': round(test_acc, 2),
            'attack_acc': round(attack_acc, 2),
            'random_guess_acc': round(100.0 / base_args.num_users, 1)
        }
        df_results = pd.concat([df_results, pd.DataFrame([new_row])], ignore_index=True)
        df_results.to_csv(results_file, index=False)
        print(f"    --> Result: Test Accuracy = {test_acc:.2f}%, SIA Attack Accuracy = {attack_acc:.2f}% (Saved to {results_file})")

    print(f"\n=======================================================")
    print(f"All experiments successfully completed!")
    print(f"Results saved to: {results_file}")
    print(f"=======================================================\n")


if __name__ == '__main__':
    main()
