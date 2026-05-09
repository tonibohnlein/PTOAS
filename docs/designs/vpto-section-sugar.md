# VPTO section sugar

## 背景

VPTO fatobj 工作流当前使用显式双 module 编程模型：

```mlir
module attributes {pto.target_arch = "a5"} {
  module attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    func.func @kernel(...) attributes {pto.aicore} {
      ...
    }
  }

  module attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
    func.func @kernel(...) attributes {pto.aicore} {
      ...
    }
  }
}
```

这个模型适合作为后端规范输入，但对手写 mixed kernel 不够紧凑。我们希望增加一个语法糖：用户可以在一个 `pto.aicore` 函数内用 `pto.section.vector` 和 `pto.section.cube` 分别编写 vector / cube 代码，VPTO 路径入口处立刻把它解包为现有双 module 形式。

## 调研结论

1. `pto.section.cube` / `pto.section.vector` 已经存在于 PTO dialect。

2. 这两个 op 当前定义在 `include/PTO/IR/PTOOps.td`，是 `SingleBlock, NoTerminator` 的 region container。

3. `PTOWrapFunctionsInSectionsPass` 已经能把带 `pto.kernel_kind` 的 frontend function body 包进对应 section，但它服务的是旧 frontend section/EmitC 模型，不是 VPTO mixed module 解包。

4. 多个 verifier 已经认识 section 上下文，例如 tpush/tpop/tfree 允许出现在 `pto.section.cube/vector` 内。

5. VPTO fatobj 后端当前只认规范双 module：
   - `vpto-normalize-container` 把单个带 `pto.kernel_kind` 的 module 包成外层 container，并要求外层只包含带 `pto.kernel_kind` 的子 module。
   - `VPTOLLVMEmitter` 按子 module 的 `pto.kernel_kind` 选择 cube/vector LLVM 目标，并给 `pto.aicore` 函数补 `_mix_aic` / `_mix_aiv` 后缀。
   - `VPTOHostStubEmission` 根据同名 `pto.aicore` 函数生成一个 host stub，并校验 mixed variants 的签名一致。

结论：新语法糖应复用现有 `pto.section.cube/vector` op，只新增 VPTO 入口解包 pass，不改变 LLVM emitter、host stub emission 和 fatobj emission 的核心模型。

## 输入形式

语法糖输入是一个普通 kernel module，module 上不需要 `pto.kernel_kind`。同一个 `pto.aicore` 函数内可以包含一个 vector section、一个 cube section，或者只包含其中一个。

```mlir
module attributes {pto.target_arch = "a5"} {
  func.func @kernel(%src: !pto.ptr<i16, gm>, %dst: !pto.ptr<i16, gm>)
      attributes {pto.aicore} {
    %c0 = arith.constant 0 : i64

    pto.section.vector {
      // vector code
    }

    pto.section.cube {
      // cube code
    }

    return
  }
}
```

section 内的代码允许使用函数参数、函数内 section 外定义的 SSA 值，以及同一 section 内定义的值。解包 pass 不单独分析这些依赖，而是整体 clone 原函数，再按目标 core 删除另一类 section。

## 输出形式

解包后的 IR 必须是现有 VPTO fatobj 规范输入：

```mlir
module attributes {pto.target_arch = "a5"} {
  module attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    func.func @kernel(%src: !pto.ptr<i16, gm>, %dst: !pto.ptr<i16, gm>)
        attributes {pto.aicore} {
      // original function body with cube sections removed
      return
    }
  }

  module attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
    func.func @kernel(%src: !pto.ptr<i16, gm>, %dst: !pto.ptr<i16, gm>)
        attributes {pto.aicore} {
      // original function body with vector sections removed
      return
    }
  }
}
```

后续 VPTO pipeline 不再感知 `pto.section.cube/vector`，只处理带 `pto.kernel_kind` 的子 module。

## 解包步骤

1. 识别 sugar module。

   如果顶层 module 已经是 container，且子 module 带 `pto.kernel_kind`，则认为输入已经是规范双 module，不做 section 解包。

   如果顶层 module 自身带 `pto.kernel_kind`，则由 `vpto-normalize-container` 包一层外层 container。

   如果顶层 module 不带 `pto.kernel_kind`，且含有 `pto.aicore` 函数内的 `pto.section.cube/vector`，则进入 section sugar 解包。

2. 为每个实际出现的 kernel kind 创建一个子 module。

   vector section 生成 `module attributes {pto.kernel_kind = #pto.kernel_kind<vector>}`。

   cube section 生成 `module attributes {pto.kernel_kind = #pto.kernel_kind<cube>}`。

   外层 module 保留 `pto.target_arch` 等 module 级公共属性。

3. 为每个带 `pto.aicore` 的函数生成同名函数 variant。

   输出函数保留原函数名、参数列表、结果类型和 `pto.aicore` 属性。后续 `VPTOLLVMEmitter` 仍负责补 `_mix_aiv` / `_mix_aic` 后缀。

   vector module 中放原函数的一个 clone，然后删除其中所有 `pto.section.cube`。

   cube module 中放原函数的一个 clone，然后删除其中所有 `pto.section.vector`。

4. 展开目标 section。

   在 vector module 中，把 `pto.section.vector` 替换为其 body 内的操作。

   在 cube module 中，把 `pto.section.cube` 替换为其 body 内的操作。

   因为每个目标函数是从原函数整体 clone 出来的，section 外公共前置代码会自然保留，不需要单独分析和克隆 section 依赖。

5. 依赖校验。

   解包 pass 不做复杂的跨 section 依赖分析。删除非目标 section 后，如果目标代码仍引用了被删除 section 产生的 SSA 值，后续 MLIR verifier 应直接报错。

   这也是期望行为：cube/vector section 之间不能通过普通 SSA 值直接传递数据，跨 core 通信必须用显式同步和搬移 op 表达。

## 约束

1. 一个 `pto.aicore` 函数内每种 section 最多出现一次。

2. `pto.section.cube` 和 `pto.section.vector` 不能嵌套。

3. section sugar 输入中，`pto.aicore` 函数 body 的顶层可包含公共前置定义、section op、同步/搬移等普通操作和 `return`。这些 section 外操作会被完整保留到每个目标函数中。

4. 同一个输入 module 中如果有多个 `pto.aicore` 函数，则每个函数只进入它实际包含的 section kind 对应子 module。后续 host stub 继续要求同名 mixed variants 的签名一致。

5. helper 函数随目标 module 一起复制。无用 helper 可以交给后续 DCE 或保持存在，不作为 section sugar 的语义问题。

6. 解包后不保留 section op。section op 只是源级 sugar，不进入 VPTO LLVM emission。

## 放置位置

新增 pass 命名为 `vpto-split-cv-module`。

它应当在 VPTO 路径最前面执行，位置早于：

1. `vpto-normalize-container`

2. `prepareVPTOForEmission`

推荐把入口职责调整为：

```text
VPTO input
  -> expand section sugar to kernel_kind modules
  -> normalize single kernel_kind module to outer container
  -> verify normalized container
  -> existing nested VPTO pipeline
  -> LLVM modules
  -> fatobj
```

这样后续 fatobj workflow 仍只有一种规范 IR 形态，不需要在 emitter 或 stub 生成阶段处理 section。

## 与现有 pass 的关系

`PTOWrapFunctionsInSectionsPass` 和本设计方向相反：

```text
kernel_kind function -> section.cube/vector
```

新 pass 的方向是：

```text
section.cube/vector -> kernel_kind module
```

因此不复用 `PTOWrapFunctionsInSectionsPass`，但可以复用它对 section op 的遍历经验和 verifier 约束。

## 测试计划

1. 添加一个 lit 测试，输入单 module + `pto.section.vector` / `pto.section.cube`，检查解包后出现两个 `pto.kernel_kind` 子 module。

2. 添加只含 vector section 的测试，检查它等价于单 vector module 输入。

3. 添加错误测试：
   - 同一个函数里重复 vector section。
   - section 嵌套。
   - section 捕获不可克隆的外部 SSA 值。
   - `pto.aicore` 函数有返回值。

4. 把现有一个 mixed VPTO host validation case 改写成 section sugar 输入，确认 `ptoas --pto-backend=vpto` 仍能直接生成 fatobj 并通过 SIM。
