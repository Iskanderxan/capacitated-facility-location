| method | candidates | status | seconds | variables | constraints | open_servers | total_cost | gap_vs_bound_% |
|---|---|---|---|---|---|---|---|---|
| greedy | k=10 | heuristic | 0.01 |  |  | 122 | 4,842,164 | 46.96 |
| mip (strong) | k=3 | within_gap | 4.42 | 2598 | 3398 | 117 | 3,395,064 | 3.04 |
| mip (strong) | k=5 | within_gap | 4.12 | 4199 | 4999 | 109 | 3,327,127 | 0.98 |
| mip (strong) | k=10 | within_gap | 7.06 | 8200 | 9000 | 108 | 3,324,776 | 0.91 |
| mip (strong) | k=20 | within_gap | 11.12 | 16200 | 17000 | 109 | 3,324,041 | 0.89 |
| mip (strong) | k=all | within_gap | 95.42 | 160200 | 161000 | 108 | 3,327,879 | 1.00 |
| mip (weak) | k=10 | within_gap | 5.04 | 8200 | 1000 | 109 | 3,323,928 | 0.88 |

Lower bound used for the gap column: 3,294,785 (full-model LP relaxation: 3,254,844).
