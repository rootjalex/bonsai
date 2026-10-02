# Bonsai syntax highlighting

A TextMate grammar for `.bonsai` files, packaged as a VS Code extension. It
highlights what the lexer in `src/Parser/Lexer.cpp` knows about -- keywords,
literals, comments, operators -- plus a few things the parser knows: builtin
types (`f32`, `vector[...]`, `array[...]`), declarations (`func`, `element`,
`tree[[T]] Name`), type positions (`x : vec3f`, `x : mut Interval`,
`l : reduce(+) vec4f`, `-> Float`), the schedule
directives (`.loopify()`, `.bind()`, ...) and the layout words (`tagged_index`,
`inline`, `tight`). There is no language server here: no completion, no errors,
no go-to-definition. Just colour.

## VS Code

```sh
editors/vscode-bonsai/install.sh
```

then **Developer: Reload Window**. The script symlinks this directory into
`~/.vscode/extensions` and, if present, `~/.vscode-server/extensions` (what
Remote-SSH and WSL use), so an edit to the grammar is picked up on the next
reload. `install.sh --uninstall` removes the links.

To build a `.vsix` instead, for installing on another machine:

```sh
npx @vscode/vsce package      # in this directory; writes bonsai-syntax-0.1.0.vsix
code --install-extension bonsai-syntax-0.1.0.vsix
```

## Other editors

`syntaxes/bonsai.tmLanguage.json` is a plain TextMate grammar, which is the
format most editors share:

- **Zed**: TextMate JSON is accepted as-is in a language extension.
- **Sublime Text**: wants the plist flavour; convert with
  `npx tmlanguage-json-to-plist` (or any JSON-to-plist tool) and drop the
  `.tmLanguage` into `Packages/User`.
- **GitHub (Linguist)**, **bat**, **Shiki**, **highlight.js via vscode-textmate**
  all consume this JSON directly.
- **Vim/Neovim** and **Emacs** do not read TextMate grammars; a `syntax/bonsai.vim`
  would be a separate twenty-line file and is not here yet.

## Scopes

The scope names follow the TextMate conventions every theme already colours
(`keyword.control`, `storage.type`, `entity.name.function`, `constant.numeric`,
...) so no theme changes are needed. The Bonsai-specific ones, for a theme
that wants to single them out:

| Scope | What it marks |
|---|---|
| `support.function.schedule.bonsai` | `.loopify`, `.bind`, `.defer`, `.sort`, `.split`, `.queue`, `.specialize`, `.vectorize`, `.skip`, `.reorder` |
| `support.constant.layout.bonsai` | `tagged_index`, `inline`, `tight` |
| `support.constant.storage.bonsai` | `Heap`, `Stack`, `DeviceGlobal`, `DeviceShared`, `Managed`, `ExternHost`, `ExternDevice`, `ExternManaged` -- `queue()`'s storage word |
| `support.function.builtin.bonsai` | `fma`, `sqrt`, `cast`, `range`, `select`, ... |
| `entity.name.type.interface.bonsai` | `IFloat`, `IVector`, any `I` + capital |
| `storage.type.tree.bonsai` | the `tree` keyword |
| `storage.modifier.reduce.bonsai` | `reduce` in a reduction variable's type, `l : reduce(+) vec4f` |
| `keyword.operator.reduce.bonsai` | the operation inside it, the `+` |
| `meta.generic.bonsai` | the inside of `[[ ... ]]` |

## Keeping it current

When a keyword is added to `Lexer::get_token_type`, add it to the matching
rule in `syntaxes/bonsai.tmLanguage.json` (`#keywords`). New builtin types go
in `#types`; new schedule directives in `#schedule-directives`. Hit
**Developer: Inspect Editor Tokens and Scopes** in VS Code to see what scope a
character got.
