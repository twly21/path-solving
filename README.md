# Group 40 Hex Path Solving

Python visualization and path-solving project for a hex-grid treasure collection problem.
![path_animation](https://github.com/user-attachments/assets/d966e89f-b499-4d18-8490-f5462cdab861)

## Overview

This project searches for a path from the entry tile to collect all treasures on a hex grid while handling:

- walls
- treasure tiles
- reward tiles
- trap tiles
- step-cost modifiers
- forced forward movement tiles

The script also:

- logs the search process to `astar_log.txt`
- prints a step-by-step summary in the console
- plots the final path on the hex grid
- saves the final plot as `hex_path_solution.png`

## Files

- `path_solving.py`: main program
- `requirements.txt`: Python dependencies
- `.gitignore`: ignores generated Python/cache/log files

## Tile Legend

- `E`: entry point
- `#`: blocked tile / wall
- `T1` to `T4`: treasures
- `⊞`: reward tile, progressively reduces energy cost
- `⊖`: trap tile, progressively increases energy cost
- `⊕`: next move costs 2 steps
- `⊠`: next move costs 0.5 steps
- `⊘`: blocking trap that cannot be passed until all treasures are collected
- `⊗`: forced-forward tile

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## How To Run

From this folder, run:

```bash
python path_solving.py
```

## Output

When the script runs successfully, it produces:

- a console summary of the chosen path
- `astar_log.txt`
- `hex_path_solution.png`
- a matplotlib window showing the grid and final route

## Notes

- The script uses a conservative search score because tile effects can change move cost dynamically.
- The original assignment file was preserved separately; this folder is a cleaned GitHub-ready version.

## Author

Primary author:
- Ives Tan Kian Hang

Development context:
- coursework / academic project for Artificial Intelligence
