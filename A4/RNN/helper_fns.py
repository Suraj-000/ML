import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import os

import re
import numpy as np
import matplotlib.pyplot as plt

def extract_between(tag, text):
    pattern = rf"<{tag}_START>(.*?)<{tag}_END>"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def parse_coords(s):
    if not s: 
        return None
    nums = re.findall(r"-?\d+", s)
    return tuple(map(int, nums)) if len(nums) == 2 else None

def plot_maze(tokens, rows=6, cols=6,show=False):
    text = " ".join(tokens) if isinstance(tokens, (list, tuple)) else str(tokens)
    adj_section = extract_between("ADJLIST", text)
    origin_section = extract_between("ORIGIN", text)
    target_section = extract_between("TARGET", text)

    m_path = re.search(
        r"<PATH_START>\s*((?:\(\s*-?\d+\s*,\s*-?\d+\s*\)\s*)+)\s*<PATH_END>",
        text,
        re.DOTALL | re.IGNORECASE
    )
    path_section = m_path.group(1) if m_path else None

    m_targetpath = re.search(
        r"<TARGETPATH_START>\s*"
        r"((?:\(\s*-?\d+\s*,\s*-?\d+\s*\)\s*)+)"  
        r"(?:<[^>]+>\s*)*"                        
        r"<TARGETPATH_END>",
        text,
        re.DOTALL | re.IGNORECASE
    )
    targetpath_section = m_targetpath.group(1) if m_targetpath else None

    # parse origin/target
    origin = parse_coords(origin_section)
    target = parse_coords(target_section)

    # parse edges
    if not adj_section:
        raise ValueError("No adjacency list found in text")
    # find pairs like "(r,c) <--> (r2,c2)" possibly separated by ; or spaces
    edge_pairs = re.findall(
        r"\(\s*-?\d+\s*,\s*-?\d+\s*\)\s*<-->\s*\(\s*-?\d+\s*,\s*-?\d+\s*\)", adj_section)
    edges = []
    for em in edge_pairs:
        coords = re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", em)
        if len(coords) == 2:
            a = parse_coords(coords[0])
            b = parse_coords(coords[1])
            edges.append((a, b))

    # Parse predicted path coordinates from PATH block
    pred_path = []
    if path_section:
        # strip any nested tags just in case
        coords = re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", path_section)
        pred_path = [parse_coords(p) for p in coords]

    # Parse true path coords from TARGETPATH block
    true_path = []
    if targetpath_section:
        coords = re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", targetpath_section)
        true_path = [parse_coords(p) for p in coords]

    # Now draw the maze (same logic as your original, with small plotting improvements)
    vertical_walls = np.ones((rows, cols + 1), dtype=bool)
    horizontal_walls = np.ones((rows + 1, cols), dtype=bool)

    for e in edges:
        (r1, c1), (r2, c2) = e
        if r1 == r2:
            c_between = min(c1, c2) + 1
            if 0 <= r1 < rows and 0 <= c_between < cols + 1:
                vertical_walls[r1, c_between] = False
        elif c1 == c2:
            r_between = min(r1, r2) + 1
            if 0 <= r_between < rows + 1 and 0 <= c1 < cols:
                horizontal_walls[r_between, c1] = False

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')

    # light grid
    for r in range(rows):
        for c in range(cols):
            x0, x1 = c, c + 1
            y_top = rows - r
            y_bot = rows - r - 1
            ax.plot([x0, x1], [y_top, y_top], color='lightgray', lw=1)
            ax.plot([x0, x1], [y_bot, y_bot], color='lightgray', lw=1)
            ax.plot([x0, x0], [y_bot, y_top], color='lightgray', lw=1)
            ax.plot([x1, x1], [y_bot, y_top], color='lightgray', lw=1)

    # draw walls
    for r in range(rows):
        for c in range(cols + 1):
            if vertical_walls[r, c]:
                x = c
                y_top = rows - r
                y_bot = rows - r - 1
                ax.plot([x, x], [y_bot, y_top], color='black', lw=4, solid_capstyle='butt')
    for r in range(rows + 1):
        for c in range(cols):
            if horizontal_walls[r, c]:
                y = rows - r
                ax.plot([c, c + 1], [y, y], color='black', lw=4, solid_capstyle='butt')

    # shade true path cells (light green)
    if true_path:
        for (r, c) in true_path:
            x0, y0 = c, rows - r - 1
            rect = plt.Rectangle((x0, y0), 1, 1, facecolor=(0.9, 1, 0.9), edgecolor=None, zorder=0)
            ax.add_patch(rect)

    # draw true path (green)
    if true_path:
        true_x = [c + 0.5 for (r, c) in true_path]
        true_y = [rows - r - 0.5 for (r, c) in true_path]
        ax.plot(true_x, true_y, linestyle='-', linewidth=3, label='True Path', zorder=5)
        ax.scatter(true_x[0], true_y[0], marker='o', s=100, zorder=6)
        ax.scatter(true_x[-1], true_y[-1], marker='s', s=100, zorder=6)

    # draw predicted path (red) with a tiny offset so overlapping segments are visible
    if pred_path:
        pred_x = [c + 0.5 + 0.08 for (r, c) in pred_path]   # small x-offset
        pred_y = [rows - r - 0.5 - 0.08 for (r, c) in pred_path]  # small y-offset
        ax.plot(pred_x, pred_y, linestyle='--', linewidth=2.5, label='Predicted Path', zorder=8)
        ax.scatter(pred_x[0], pred_y[0], marker='o', s=90, zorder=9)
        ax.scatter(pred_x[-1], pred_y[-1], marker='x', s=90, zorder=9)

    # mark origin/target if needed
    if origin:
        ox, oy = origin[1] + 0.5, rows - origin[0] - 0.5
        ax.scatter(ox, oy, c='blue', s=80, marker='o', zorder=7)
    if target:
        tx, ty = target[1] + 0.5, rows - target[0] - 0.5
        ax.scatter(tx, ty, c='blue', s=80, marker='x', zorder=7)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(cols + 1)[:-1])
    ax.set_yticks(np.arange(rows + 1)[:-1])
    plt.yticks([])
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    if show: plt.show()
    return fig

class Logger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def plot_ep_rewards_vs_iterations(episode_rewards,algo_name,path):
    plt.figure(figsize=(14,12))
    plt.plot(episode_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title(f'{algo_name} Episode Rewards ')
    plt.savefig(f'{path}')
    # plt.show()

def plot_ep_rewards_vs_iterations2(dqn, ddqn, path, window=50):
    plt.figure(figsize=(14,12))

    plt.plot(dqn, alpha=0.5, label='DQN (raw)')
    plt.plot(ddqn, alpha=0.5, label='DDQN (raw)')

    dqn_mean = np.convolve(dqn, np.ones(window)/window, mode='valid')
    ddqn_mean = np.convolve(ddqn, np.ones(window)/window, mode='valid')

    plt.plot(dqn_mean, linewidth=2, label=f'DQN {window}-episode mean')
    plt.plot(ddqn_mean, linewidth=2, label=f'DDQN {window}-episode mean')

    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Episode Rewards Comparison: DQN vs DDQN')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    # plt.show()

def print_star(n=100):
    print("*"*n)

def set_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available(): 
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def network_details(network,input_dim,output_dim,hidden_dim):
    net=network(input_dim,output_dim,hidden_dim)
    print(net)
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    # print(f"Trainable parameters: {trainable_params}")

def plot_training_curves(
        train_loss,val_loss,
        train_seq_acc,val_seq_acc,
        train_f1s,val_f1s,
        train_tok_accs,val_tok_accs,
        save_path='plots/training_curves.png'
):
    if not os.path.exists('plots'):
        os.makedirs('plots')
    epochs_range = range(len(train_loss))

    fig,axs=plt.subplots(2,2,figsize=(18,14),sharex=True)
    fig.suptitle("Training Curves",fontsize=22,fontweight="bold")
    plt.rcParams['axes.titleweight']='bold'
    colors = plt.cm.tab10.colors

    plt.tight_layout(pad=4)

    for ax in axs.flatten():
        ax.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.9)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.9)
        ax.minorticks_on()

    def style_plot(ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.25, linestyle="--")
        for spine in ax.spines.values():
            spine.set_alpha(0.1)

    ax=axs[0,0]
    ax.plot(epochs_range, train_loss, label='Train Loss',lw=2.5, marker='o', alpha=0.8, color=colors[0])
    ax.plot(epochs_range, val_loss, label='Val Loss',lw=2.5, marker='s', alpha=0.8, color=colors[1])
    style_plot(ax, "Loss Over Epochs", "Epochs", "Loss")
    ax.legend()

    ax=axs[0,1]
    ax.plot(epochs_range, train_tok_accs, label='Train Token Acc',lw=2.5, marker='o', alpha=0.8, color=colors[2])
    ax.plot(epochs_range, val_tok_accs, label='Val Token Acc',lw=2.5, marker='s', alpha=0.8, color=colors[3])
    style_plot(ax, "Token Accuracy", "Epochs", "Accuracy")
    ax.legend()

    ax = axs[1, 0]
    ax.plot(epochs_range, train_seq_acc, label="Train seq Acc", lw=2.5, marker='o', alpha=0.8, color=colors[4])
    ax.plot(epochs_range, val_seq_acc,   label="Val seq acc",   lw=2.5, marker='s', alpha=0.8, color=colors[5])
    style_plot(ax, "Sequence Accuracy", "Epochs", "Accuracy")
    ax.legend()

    # (4) F1 Score
    ax = axs[1, 1]
    ax.plot(epochs_range, train_f1s, label="Train f1", lw=2.5, marker='o', alpha=0.8, color=colors[6])
    ax.plot(epochs_range, val_f1s,   label="Val f1",   lw=2.5, marker='s', alpha=0.8, color=colors[7])
    style_plot(ax, "F1 Score", "Epochs", "F1")
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {save_path}")
