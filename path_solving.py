from itertools import count
from pathlib import Path
import heapq

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ENTRY_TILE = "E"
WALL_TILE = "#"
REDUCE_ENERGY_TILE = "⊞"
DOUBLE_STEP_TILE = "⊕"
BLOCKING_TRAP_TILE = "⊘"
PROGRESSIVE_TRAP_TILE = "⊖"
HALF_STEP_TILE = "⊠"
FORCE_FORWARD_TILE = "⊗"

HEX_MAP = {
    (0, 0): ".", (0, 1): ".", (0, 2): "#", (0, 3): ".", (0, 4): ".", (0, 5): "E",
    (1, 0): ".", (1, 1): ".", (1, 2): "⊞", (1, 3): ".", (1, 4): "⊕", (1, 5): ".",
    (2, 0): ".", (2, 1): "⊕", (2, 2): ".", (2, 3): "#", (2, 4): ".", (2, 5): ".",
    (3, 0): ".", (3, 1): "T4", (3, 2): "#", (3, 3): ".", (3, 4): "⊘", (3, 5): ".",
    (4, 0): ".", (4, 1): "#", (4, 2): ".", (4, 3): "#", (4, 4): "T1", (4, 5): "⊞",
    (5, 0): "⊠", (5, 1): ".", (5, 2): "⊗", (5, 3): ".", (5, 4): ".", (5, 5): ".",
    (6, 0): ".", (6, 1): "#", (6, 2): "#", (6, 3): ".", (6, 4): "⊗", (6, 5): ".",
    (7, 0): ".", (7, 1): "#", (7, 2): "T2", (7, 3): "⊠", (7, 4): ".", (7, 5): ".",
    (8, 0): ".", (8, 1): ".", (8, 2): ".", (8, 3): "⊖", (8, 4): "#", (8, 5): ".",
    (9, 0): ".", (9, 1): ".", (9, 2): "T3", (9, 3): ".", (9, 4): ".", (9, 5): ".",
}

ENTRY = next(position for position, value in HEX_MAP.items() if value == ENTRY_TILE)
TREASURES = {position for position, value in HEX_MAP.items() if value.startswith("T")}
OUTPUT_DIR = Path(__file__).resolve().parent
LOG_PATH = OUTPUT_DIR / "astar_log.txt"
PLOT_PATH = OUTPUT_DIR / "hex_path_solution.png"


def get_hex_neighbors(q, r):
    if q % 2 == 0:
        directions = [(0, -1), (0, 1), (1, 0), (1, -1), (-1, 0), (-1, -1)]
    else:
        directions = [(0, -1), (0, 1), (1, 1), (1, 0), (-1, 1), (-1, 0)]

    neighbors = []
    for dq, dr in directions:
        neighbor = (q + dq, r + dr)
        if neighbor in HEX_MAP and HEX_MAP[neighbor] != WALL_TILE:
            neighbors.append((neighbor, (dq, dr)))
    return neighbors


def calculate_progressive_energy(base_multiplier, reward_active, steps_after_reward, trap_active, steps_after_trap):
    energy = base_multiplier

    if reward_active and steps_after_reward > 0:
        energy /= 2 ** steps_after_reward

    if trap_active and steps_after_trap > 0:
        energy *= 2 ** steps_after_trap

    return energy


def solve_hex_path(log_path=LOG_PATH):
    frontier = []
    tie_breaker = count()
    heapq.heappush(
        frontier,
        (
            0.0,
            next(tie_breaker),
            ENTRY,
            frozenset(),
            [ENTRY],
            1.0,
            0.0,
            0.0,
            False,
            False,
            frozenset(),
            False,
            -1,
            False,
            -1,
        ),
    )
    visited = {}
    iteration = 0

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("A* Pathfinding Log\n")
        log_file.write("=" * 60 + "\n")

        while frontier:
            iteration += 1
            (
                f_score,
                _,
                current,
                collected,
                path,
                base_multiplier,
                g_energy,
                g_step,
                double_next,
                half_next,
                activated_tiles,
                reward_active,
                reward_step_index,
                trap_active,
                trap_step_index,
            ) = heapq.heappop(frontier)

            state_key = (
                current,
                collected,
                base_multiplier,
                double_next,
                half_next,
                activated_tiles,
                reward_active,
                len(path) - reward_step_index if reward_active else -1,
                trap_active,
                len(path) - trap_step_index if trap_active else -1,
            )

            if state_key in visited and visited[state_key] <= g_energy:
                continue
            visited[state_key] = g_energy

            log_file.write(
                f"Iteration {iteration}: pos={current}, collected={sorted(collected)}, f={f_score:.4f}, "
                f"energy={g_energy:.4f}, steps={g_step:.2f}\n"
            )

            tile = HEX_MAP[current]
            new_collected = set(collected)
            new_activated_tiles = set(activated_tiles)
            new_reward_active = reward_active
            new_reward_step_index = reward_step_index
            new_trap_active = trap_active
            new_trap_step_index = trap_step_index

            if tile.startswith("T"):
                new_collected.add(current)
                new_activated_tiles.add(current)

            if tile == PROGRESSIVE_TRAP_TILE and current not in new_activated_tiles:
                new_trap_active = True
                new_trap_step_index = len(path) - 1
                new_activated_tiles.add(current)

            if tile == REDUCE_ENERGY_TILE and current not in new_activated_tiles:
                new_reward_active = True
                new_reward_step_index = len(path) - 1
                new_activated_tiles.add(current)

            if tile == BLOCKING_TRAP_TILE and TREASURES - new_collected:
                log_file.write("  Blocking trap prevents progress until all treasures are collected.\n")
                continue

            if frozenset(new_collected) == TREASURES:
                log_file.write(f"Goal reached. Final path: {path}\n")
                return path

            for neighbor, direction in get_hex_neighbors(*current):
                steps_after_reward = len(path) - new_reward_step_index if new_reward_active else 0
                steps_after_trap = len(path) - new_trap_step_index if new_trap_active else 0

                move_energy = calculate_progressive_energy(
                    base_multiplier,
                    new_reward_active,
                    steps_after_reward,
                    new_trap_active,
                    steps_after_trap,
                )
                move_step = 1.0

                if current in activated_tiles:
                    move_energy = 1.0

                if double_next:
                    move_step = 2.0
                elif half_next:
                    move_step = 0.5

                next_path = path + [neighbor]
                final_neighbor = neighbor
                next_tile = HEX_MAP[neighbor]
                new_g_energy = g_energy + move_energy
                new_g_step = g_step + move_step

                if next_tile == FORCE_FORWARD_TILE:
                    step1 = (neighbor[0] + direction[0], neighbor[1] + direction[1])
                    step2 = (neighbor[0] + 2 * direction[0], neighbor[1] + 2 * direction[1])

                    if not (
                        step1 in HEX_MAP
                        and HEX_MAP[step1] != WALL_TILE
                        and step2 in HEX_MAP
                        and HEX_MAP[step2] != WALL_TILE
                    ):
                        log_file.write(f"  Skipped invalid force-forward transition from {neighbor}.\n")
                        continue

                    next_path += [step1, step2]
                    final_neighbor = step2

                    for extra_step in (1, 2):
                        extra_steps_after_reward = (len(path) + extra_step - new_reward_step_index) if new_reward_active else 0
                        extra_steps_after_trap = (len(path) + extra_step - new_trap_step_index) if new_trap_active else 0
                        new_g_energy += calculate_progressive_energy(
                            base_multiplier,
                            new_reward_active,
                            extra_steps_after_reward,
                            new_trap_active,
                            extra_steps_after_trap,
                        )

                    new_g_step += 2.0

                heuristic = 0.0
                new_f_score = new_g_energy + new_g_step + heuristic

                heapq.heappush(
                    frontier,
                    (
                        new_f_score,
                        next(tie_breaker),
                        final_neighbor,
                        frozenset(new_collected),
                        next_path,
                        base_multiplier,
                        new_g_energy,
                        new_g_step,
                        HEX_MAP[final_neighbor] == DOUBLE_STEP_TILE and final_neighbor not in new_activated_tiles,
                        HEX_MAP[final_neighbor] == HALF_STEP_TILE and final_neighbor not in new_activated_tiles,
                        frozenset(new_activated_tiles),
                        new_reward_active,
                        new_reward_step_index,
                        new_trap_active,
                        new_trap_step_index,
                    ),
                )

        log_file.write("No path found.\n")
        return None


def summarize_path(path):
    if not path:
        return None

    base_multiplier = 1.0
    double_next = False
    half_next = False
    energy_sum = 0.0
    step_sum = 0.0
    seen_tiles = set()
    reward_active = False
    reward_step_index = -1
    trap_active = False
    trap_step_index = -1

    activated_treasures = []
    activated_traps = []
    activated_rewards = []
    activated_specials = []
    step_costs = [(path[0], 0.0, 0.0, 0.0, 0.0)]
    log_lines = [f"Start: {path[0]} (Step 0)"]

    for step_idx, current_pos in enumerate(path):
        tile = HEX_MAP[current_pos]

        if current_pos not in seen_tiles:
            if tile.startswith("T"):
                activated_treasures.append((current_pos, tile, step_idx))
                seen_tiles.add(current_pos)
                log_lines.append(f"Step {step_idx}: Collected {tile} at {current_pos}")
            elif tile == BLOCKING_TRAP_TILE:
                activated_traps.append((current_pos, "Blocking Trap", step_idx))
                seen_tiles.add(current_pos)
                log_lines.append(f"Step {step_idx}: Encountered blocking trap at {current_pos}")
            elif tile == PROGRESSIVE_TRAP_TILE:
                trap_active = True
                trap_step_index = step_idx
                activated_rewards.append((current_pos, "Trap 1", step_idx, "Progressive energy doubling starts"))
                seen_tiles.add(current_pos)
                log_lines.append(f"Step {step_idx}: Activated Trap 1 at {current_pos}")
            elif tile == REDUCE_ENERGY_TILE:
                reward_active = True
                reward_step_index = step_idx
                activated_rewards.append((current_pos, "Reward 1", step_idx, "Progressive energy reduction starts"))
                seen_tiles.add(current_pos)
                log_lines.append(f"Step {step_idx}: Activated Reward 1 at {current_pos}")
            elif tile == DOUBLE_STEP_TILE:
                double_next = True
                activated_rewards.append((current_pos, "Step Doubler", step_idx, "Next move costs 2 steps"))
                seen_tiles.add(current_pos)
                log_lines.append(f"Step {step_idx}: Activated step doubler at {current_pos}")
            elif tile == HALF_STEP_TILE:
                half_next = True
                activated_rewards.append((current_pos, "Step Halver", step_idx, "Next move costs 0.5 steps"))
                seen_tiles.add(current_pos)
                log_lines.append(f"Step {step_idx}: Activated step halver at {current_pos}")
            elif tile == FORCE_FORWARD_TILE:
                activated_specials.append((current_pos, "Force Forward", step_idx))
                seen_tiles.add(current_pos)
                log_lines.append(f"Step {step_idx}: Triggered force-forward tile at {current_pos}")

        if step_idx >= len(path) - 1:
            continue

        next_pos = path[step_idx + 1]
        steps_after_reward = step_idx - reward_step_index if reward_active else 0
        steps_after_trap = step_idx - trap_step_index if trap_active else 0

        energy = calculate_progressive_energy(
            base_multiplier,
            reward_active,
            steps_after_reward,
            trap_active,
            steps_after_trap,
        )

        if half_next:
            step_cost = 0.5
            half_next = False
        elif double_next:
            step_cost = 2.0
            double_next = False
        else:
            step_cost = 1.0

        energy_sum += energy
        step_sum += step_cost
        step_costs.append((next_pos, round(energy, 4), round(energy_sum, 4), round(step_cost, 2), round(step_sum, 2)))
        log_lines.append(
            f"Move {step_idx} -> {step_idx + 1}: {current_pos} -> {next_pos} | "
            f"Energy {energy:.4f} | Total Energy {energy_sum:.4f} | "
            f"Step {step_cost:.2f} | Total Steps {step_sum:.2f}"
        )

    summary = {
        "path": path,
        "energy_sum": round(energy_sum, 4),
        "step_sum": round(step_sum, 2),
        "activated_treasures": activated_treasures,
        "activated_traps": activated_traps,
        "activated_rewards": activated_rewards,
        "activated_specials": activated_specials,
        "step_costs": step_costs,
        "log_lines": log_lines,
    }
    return summary


def print_summary(summary):
    print("Hex Pathfinding Summary")
    print("=" * 80)
    for line in summary["log_lines"]:
        print(line)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"End Position: {summary['path'][-1]} (Step {len(summary['path']) - 1})")
    print(f"Total Energy Cost: {summary['energy_sum']:.4f}")
    print(f"Total Step Cost: {summary['step_sum']:.2f}")
    print(f"Path Length: {len(summary['path'])} positions")

    print(f"\nTreasures Collected ({len(summary['activated_treasures'])}):")
    for pos, tile, step in summary["activated_treasures"]:
        print(f"  {tile} at {pos} (Step {step})")

    print(f"\nRewards Activated ({len(summary['activated_rewards'])}):")
    for pos, reward_type, step, details in summary["activated_rewards"]:
        print(f"  {reward_type} at {pos} (Step {step}) - {details}")

    if summary["activated_traps"]:
        print(f"\nTraps Encountered ({len(summary['activated_traps'])}):")
        for pos, trap_type, step in summary["activated_traps"]:
            print(f"  {trap_type} at {pos} (Step {step})")

    if summary["activated_specials"]:
        print(f"\nSpecial Tiles Triggered ({len(summary['activated_specials'])}):")
        for pos, special_type, step in summary["activated_specials"]:
            print(f"  {special_type} at {pos} (Step {step})")


def draw_hex(center, size, color):
    return patches.RegularPolygon(
        center,
        numVertices=6,
        radius=size,
        orientation=np.pi / 6,
        facecolor=color,
        edgecolor="black",
    )


def plot_path(path, step_costs, output_path=PLOT_PATH):
    colors = {
        ".": "white",
        "#": "#6A6A6A",
        "T1": "#FFA200",
        "T2": "#FFA200",
        "T3": "#FFA200",
        "T4": "#FFA200",
        PROGRESSIVE_TRAP_TILE: "#CD73E4",
        BLOCKING_TRAP_TILE: "#CD73E4",
        FORCE_FORWARD_TILE: "#CD73E4",
        DOUBLE_STEP_TILE: "#CD73E4",
        REDUCE_ENERGY_TILE: "#37C4BD",
        HALF_STEP_TILE: "#37C4BD",
        ENTRY_TILE: "#6260F6",
    }

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect("equal")
    ax.axis("off")

    coord_map = {}
    size = 1.0
    for (q, r), symbol in HEX_MAP.items():
        x = 1.5 * q
        y = np.sqrt(3) * (r + 0.5 * (q % 2))
        coord_map[(q, r)] = (x, y)
        ax.add_patch(draw_hex((x, y), size, colors.get(symbol, "white")))
        ax.text(x, y + 0.2, f"{symbol}", ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x, y - 0.5, f"({q},{r})", ha="center", va="center", fontsize=7)

    for start, end in zip(path, path[1:]):
        start_x, start_y = coord_map[start]
        end_x, end_y = coord_map[end]
        ax.arrow(
            start_x,
            start_y,
            end_x - start_x,
            end_y - start_y,
            head_width=0.2,
            head_length=0.2,
            fc="blue",
            ec="blue",
            length_includes_head=True,
        )

    for pos, _, total_energy, _, total_steps in step_costs:
        x_coord, y_coord = coord_map[pos]
        ax.text(x_coord + 0.1, y_coord + 0.4, f"E:{total_energy}\nS:{total_steps}", color="red", fontsize=8)

    plt.title("Hex Grid Path Solution")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()


def main():
    path = solve_hex_path()
    if not path:
        print("No valid path was found.")
        return

    summary = summarize_path(path)
    print_summary(summary)
    plot_path(path, summary["step_costs"])
    print(f"\nLog written to: {LOG_PATH}")
    print(f"Plot saved to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
