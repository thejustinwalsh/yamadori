| arm | stage | prompt_n | prompt tok/s | decode tok/s | draft acc | pool max+1 | used |
|---|---|---|---|---|---|---|---|
| current | A_deep_slot0 | 160000 | 464 | 19.86 | 186/272 | 160249 | 160249 |
| current | B_new_slot1_while_slot0_holds | 60000 | 233 | 9.20 | 116/224 | 220426 | 220426 |
| current | C_clear_slot0 | 1 | 3 | 0.00 | - | 220426 | 60171 |
| current | D_slot1_continues_after_clear | 257 | 162 | 7.75 | 121/312 | 220427 | 60628 |
| current | E_clear_slot1 | 1 | 16 | 0.00 | - | 1 | 1 |
| current | E_same_prompt_from_empty_pool_slot2 | 60000 | 757 | 64.11 | 106/150 | 60182 | 60182 |
| current | F_clear_slot2 | 1 | 20 | 0.00 | - | 2 | 2 |
| current | F_idle_40000_slot0 | 39999 | 837 | 59.41 | 4/4 | 40005 | 40005 |
| current | F_clear_slot1_before_8000 | 1 | 32 | 0.00 | - | 40008 | 40007 |
| current | F_active_8000_slot1 | 7999 | 659 | 58.55 | 136/238 | 48260 | 48260 |
| current | F_clear_slot1_before_32000 | 1 | 19 | 0.00 | - | 48008 | 40008 |
| current | F_active_32000_slot1 | 31999 | 602 | 54.04 | 134/242 | 72260 | 72260 |
| current | F_clear_slot1_before_64000 | 1 | 14 | 0.00 | - | 48008 | 40008 |
| current | F_active_64000_slot1 | 59999 | 543 | 55.83 | 105/152 | 100188 | 100188 |
