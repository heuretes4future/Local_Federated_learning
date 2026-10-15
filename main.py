import torch
import time
from config import server_config 
from model import CNN, run_network
from aggregation import FedAvg
from dataset import dataset_loader, label_skew_split
from adaptation import adaptation, cosine_similarity, local_adaptation, set_seed
from torch.utils.data import DataLoader
from evaluation import evaluate_global, evaluate_per_client, summarise, summarise_stability
import pandas as pd
import os

if __name__ == "__main__":

    test_dataset, train_dataset = dataset_loader()
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
    results = []

    # Save CSVs next to this script, not wherever Python was launched from
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for seed in server_config['seeds']:
        set_seed(seed)

        for curr in server_config['strategies']:

            for skew in server_config['skew_levels']:
                client_lrs = [server_config['base_lr']] * server_config['N_clients']
                client_loss_history = [[] for _ in range(server_config['N_clients'])]

                # FIX: re-seed before each model init so weights are reproducible per condition
                set_seed(seed)
                global_weights = CNN().state_dict()

                current_dataset_skew = [
                    label_skew_split(
                        train_dataset,
                        client_id=cid,
                        client_amount=server_config['N_clients'],
                        classes_per_client=skew,
                        seed=seed
                    )
                    for cid in range(server_config['N_clients'])
                ]

                for round_num in range(server_config['comm_rounds']):
                    t0 = time.time()

                    thread_results = [
                        run_network(current_dataset_skew[cid], global_weights, client_lrs[cid])
                        for cid in range(server_config['N_clients'])
                    ]

                    all_weights = [r[0] for r in thread_results]
                    all_losses  = [r[1] for r in thread_results]

                    for cid in range(server_config['N_clients']):
                        client_loss_history[cid].append(all_losses[cid][0])

                    t1 = time.time()
                    client_sizes = [len(ds) for ds in current_dataset_skew]

                    # FIX: capture pre-FedAvg global weights here, AFTER client training,
                    # so SGA compares client updates against the correct current global model
                    pre_aggregation_global = {k: v.clone() for k, v in global_weights.items()}

                    global_weights = FedAvg(all_weights, client_sizes)
                    t2 = time.time()

                    if curr == 'LA':
                        for cid in range(server_config['N_clients']):
                            client_lrs[cid] = local_adaptation(client_lrs[cid], client_loss_history[cid])

                    elif curr == 'SGA':
                        if round_num % server_config['lr_step_rounds'] == 0:
                            # FIX: pass pre_aggregation_global (the model clients actually trained from)
                            increasing, decreasing = cosine_similarity(all_weights, pre_aggregation_global)
                            for cid in range(len(all_weights)):
                                client_lrs[cid] = adaptation(client_lrs[cid], increasing, decreasing)

                    eval_rounds = [0, 10, 25, 50, 75, 99]
                    if round_num in eval_rounds:
                        global_acc, global_loss = evaluate_global(global_weights, test_loader)
                        client_evals = evaluate_per_client(all_weights, current_dataset_skew)

                        avg_client_acc  = sum(e['accuracy'] for e in client_evals) / len(client_evals)
                        avg_client_loss = sum(e['loss'] for e in client_evals) / len(client_evals)

                        results.append({
                            'seed': seed,
                            'strategy': curr,
                            'skew': skew,
                            'round': round_num,
                            'global_acc': global_acc,
                            'global_loss': global_loss,
                            'avg_client_acc': avg_client_acc,
                            'avg_client_loss': avg_client_loss,
                        })

                        print(
                            f"RESULT | seed: {seed} | strategy: {curr} | classes per client: {skew} | round: {round_num} | "
                            f"global acc: {global_acc:.4f} | global loss: {global_loss:.4f} | "
                            f"avg client acc: {avg_client_acc:.4f} | avg client loss: {avg_client_loss:.4f}",
                            flush=True
                        )

                        pd.DataFrame(results).to_csv(os.path.join(script_dir, "results_partial.csv"), index=False)

                    if round_num % server_config['print_every'] == 0 or round_num == server_config['comm_rounds'] - 1:
                        print(
                            f"PROGRESS | seed: {seed} | strategy: {curr} | classes per client: {skew} | "
                            f"round: {round_num}/{server_config['comm_rounds'] - 1} | "
                            f"training: {t1-t0:.1f}s | FedAvg: {t2-t1:.1f}s | "
                            f"lr range: {min(client_lrs):.5f}-{max(client_lrs):.5f}",
                            flush=True
                        )

    summarise(results)
    stability_results = summarise_stability(results)
    pd.DataFrame(results).to_csv(os.path.join(script_dir, "results.csv"), index=False)
    pd.DataFrame(stability_results).to_csv(os.path.join(script_dir, "stability_results.csv"), index=False)
    print(f"\nFiles saved to: {script_dir}")
