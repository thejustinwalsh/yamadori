| arm | stage | prompt_n | prompt tok/s | decode tok/s | draft acc | pool max+1 | used |
|---|---|---|---|---|---|---|---|
| kvplace2 | A_deep_slot0 | 160000 | 465 | 20.02 | 186/272 | 160249 | 160249 |
| kvplace2 | B_new_slot1_while_slot0_holds | 60000 | 667 | 62.47 | 106/150 | 220763 | 220435 |
| kvplace2 | C_clear_slot0 | 1 | 10 | 0.00 | - | 60180 | 60180 |
| kvplace2 | D_slot1_continues_after_clear | 257 | 402 | 59.98 | 142/225 | 60692 | 60692 |
| kvplace2 | E_clear_slot1 | 1 | 16 | 0.00 | - | 60181 | 1 |
| kvplace2 | E_same_prompt_from_empty_pool_slot2 | 60000 | 757 | 63.93 | 106/150 | 60182 | 60182 |
| kvplace2 | F_clear_slot2 | 1 | 20 | 0.00 | - | 60181 | 2 |
| kvplace2 | F_idle_40000_slot0 | 39999 | 508 | 50.57 | 4/4 | 100183 | 40005 |
| kvplace2 | F_clear_slot1_before_8000 | 1 | 36 | 0.00 | - | 100186 | 40007 |
| kvplace2 | F_active_8000_slot1 | 7999 | 1000 | 70.03 | 141/226 | 100187 | 48260 |
| kvplace2 | F_clear_slot1_before_32000 | 1 | 27 | 0.00 | - | 100187 | 40008 |
| kvplace2 | F_active_32000_slot1 | 31999 | 876 | 61.95 | 135/240 | 100188 | 72260 |
| kvplace2 | F_clear_slot1_before_64000 | 1 | 20 | 0.00 | - | 100187 | 40008 |
| kvplace2 | F_active_64000_slot1 | 59999 | 756 | 63.04 | 106/150 | 100188 | 100188 |
| kvplace2 | G_concurrent_slots_1_2 | None | 0 | 18.02 | - | 100187 | 56515 |
| kvplace2 | G_clear_slot1 | 1 | 26 | 0.00 | - | 100187 | 48262 |
| kvplace2 | G_alone_slot1 | 7999 | 999 | 64.39 | 136/238 | 100188 | 56514 |
| kvplace2 | G_clear_slot2 | 1 | 20 | 0.00 | - | 100187 | 48259 |
| kvplace2 | G_alone_slot2 | 7999 | 908 | 83.18 | 164/182 | 100188 | 56514 |
