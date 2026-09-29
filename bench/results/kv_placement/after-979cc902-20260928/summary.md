| arm | stage | prompt_n | prompt tok/s | decode tok/s | draft acc | pool max+1 | used |
|---|---|---|---|---|---|---|---|
| kvplace | A_deep_slot0 | 160000 | 470 | 19.67 | 186/272 | 160249 | 160249 |
| kvplace | B_new_slot1_while_slot0_holds | 60000 | 671 | 60.14 | 106/150 | 262144 | 220435 |
| kvplace | C_clear_slot0 | 1 | 10 | 0.00 | - | 60180 | 60180 |
| kvplace | D_slot1_continues_after_clear | 257 | 390 | 58.64 | 140/228 | 60691 | 60691 |
| kvplace | E_clear_slot1 | 1 | 16 | 0.00 | - | 60181 | 1 |
| kvplace | E_same_prompt_from_empty_pool_slot2 | 60000 | 758 | 63.11 | 100/144 | 262144 | 60173 |
| kvplace | F_clear_slot2 | 1 | 19 | 0.00 | - | 262144 | 2 |
| kvplace | F_idle_40000_slot0 | 39999 | 837 | 59.68 | 4/4 | 262144 | 40005 |
| kvplace | F_clear_slot1_before_8000 | 1 | 32 | 0.00 | - | 40007 | 40007 |
| kvplace | F_active_8000_slot1 | 7999 | 853 | 68.41 | 138/233 | 262144 | 48261 |
| kvplace | F_clear_slot1_before_32000 | 1 | 25 | 0.00 | - | 262144 | 40008 |
| kvplace | F_active_32000_slot1 | 31999 | 857 | 61.41 | 134/241 | 262144 | 72261 |
| kvplace | F_clear_slot1_before_64000 | 1 | 23 | 0.00 | - | 262144 | 40008 |
| kvplace | F_active_64000_slot1 | 59999 | 753 | 63.46 | 105/152 | 262144 | 100188 |
| kvplace | G_concurrent_slots_1_2 | None | 0 | 18.15 | - | 262144 | 56515 |
| kvplace | G_clear_slot1 | 1 | 26 | 0.00 | - | 262144 | 48262 |
| kvplace | G_alone_slot1 | 7999 | 1000 | 67.01 | 137/236 | 262144 | 56514 |
| kvplace | G_clear_slot2 | 1 | 23 | 0.00 | - | 262144 | 48259 |
| kvplace | G_alone_slot2 | 7999 | 963 | 80.03 | 156/197 | 262144 | 56515 |
