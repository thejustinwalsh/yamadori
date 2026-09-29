| arm | stage | prompt_n | prompt tok/s | decode tok/s | draft acc | pool max+1 | used |
|---|---|---|---|---|---|---|---|
| dev | A_deep_slot0 | 160000 | 467 | 19.98 | 186/272 | 160249 | 160249 |
| dev | B_new_slot1_while_slot0_holds | 60000 | 669 | 60.24 | 106/150 | 262144 | 220435 |
| dev | C_clear_slot0 | 1 | 11 | 0.00 | - | 60180 | 60180 |
| dev | D_slot1_continues_after_clear | 257 | 404 | 58.98 | 140/228 | 60691 | 60691 |
| dev | E_clear_slot1 | 1 | 16 | 0.00 | - | 60181 | 1 |
| dev | E_same_prompt_from_empty_pool_slot2 | 60000 | 757 | 63.58 | 100/144 | 262144 | 60173 |
| dev | F_clear_slot2 | 1 | 20 | 0.00 | - | 262144 | 2 |
| dev | F_idle_40000_slot0 | 39999 | 836 | 60.07 | 4/4 | 262144 | 40005 |
| dev | F_clear_slot1_before_8000 | 1 | 32 | 0.00 | - | 40007 | 40007 |
| dev | F_active_8000_slot1 | 7999 | 854 | 67.97 | 138/233 | 262144 | 48261 |
| dev | F_clear_slot1_before_32000 | 1 | 24 | 0.00 | - | 262144 | 40008 |
| dev | F_active_32000_slot1 | 31999 | 856 | 61.24 | 134/241 | 262144 | 72261 |
| dev | F_clear_slot1_before_64000 | 1 | 22 | 0.00 | - | 262144 | 40008 |
| dev | F_active_64000_slot1 | 59999 | 752 | 62.92 | 105/152 | 262144 | 100188 |
