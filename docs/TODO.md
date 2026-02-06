1. PerfCheck c++ vs dsl (including fwd and bwd)
2. dsl 反向代码现在 all mask？精细控制下？
3. 反向是否需要改为这种格式？ (避免 q 越界的部分，也使用 arbitrary mask)
for mask block
  is first (apply mask_seq)
  not first (just mask)
for full block
  is first (apply mask_seq)
  not first (non mask)
4. 编译过于耗时，如何加速编译 (把 casual local 提出来？) （如果 dsl ok，优先级低）
5. PackGQA存在 error，在 PackGQA 的模式下，arbitrary Func 也应该使用 PackGQA 的空间，即q0h0 q0h1 q0h2... q1h0 q1h1 q1h2...？
6. dsl Hopper bwd check dim=256？ 是否需要 merge 最新的 dsl 代码？
7. dsl Hopper bwd 目前不支持 gqa mqa


