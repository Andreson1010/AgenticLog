# Features — especificações por feature

Este diretório guarda o **planejamento técnico** de cada feature do AgenticLog. Os arquivos aqui são o contrato aprovado **antes** de qualquer mudança em `src/`, `app.py` ou `tests/` de implementação.

O pipeline [feature-factory](../../.claude/skills/feature-factory/SKILL.md) (Fase 3 + Checkpoint 2) e o agent `spec-writer` são responsáveis por criar e revisar estes artefatos. A implementação continua nas fases `backend-builder` e `frontend-builder` — **não** usa TLC Execute.

## Estrutura

```
.specs/features/
├── README.md                    ← este arquivo
└── <feature-slug>/              ← um diretório por feature (kebab-case)
    ├── spec.md                  ← sempre (escopo medium, large ou complex)
    ├── design.md                ← opcional: large / complex
    ├── tasks.md                 ← opcional: large / complex
    └── context.md               ← opcional: decisões de discuss (TLC) se houver ambiguidade
```

### Convenção de nome (`feature-slug`)

- **kebab-case**, curto e descritivo: `lazy-llm-init`, `retry-llm-backoff`
- Um slug por feature; não reutilize pasta para features diferentes
- Prefixo de requirement IDs na spec: derivado do slug (ex.: `LAZY-01`, `RETRY-02`)

## Arquivos por escopo TLC

O orquestrador classifica cada feature no início da Fase 3:

| Escopo | Quando usar | Arquivos |
|--------|-------------|----------|
| **medium** | Feature clara, menos de 10 passos de implementação, sem padrão arquitetural novo | `spec.md` |
| **large** | Vários componentes, interações novas ou decisões de arquitetura explícitas | `spec.md` + `design.md` + `tasks.md` |
| **complex** | Domínio novo, alta ambiguidade ou riscos graves em `CONCERNS.md` | Idem large; resolver open questions antes do Checkpoint 2 |

Templates e detalhes: skill global **`tlc-spec-driven`** (`references/specify.md`, `design.md`, `tasks.md`), estendidos pelas seções técnicas em `.claude/agents/spec-writer.md`.

## O que vai em cada arquivo

| Arquivo | Conteúdo |
|---------|----------|
| **spec.md** | Problema, goals, out of scope, user stories (WHEN/THEN/SHALL), traceability IDs, data model, API, frontend, testes exigidos, arquivos que mudam, riscos, open questions, success criteria |
| **design.md** | Arquitetura, componentes, interfaces, reuso de código, diagramas (mermaid opcional), mitigação de itens de `CONCERNS.md` |
| **tasks.md** | Tarefas atômicas com dependências, IDs ligados à spec, critérios “Done when” — **plano para builders**, não substitui o feature-factory |
| **context.md** | Decisões humanas em áreas cinzentas (só se necessário) |

## Relação com o resto de `.specs/`

| Pasta | Papel |
|-------|--------|
| [.specs/project/](../project/) | Visão (`PROJECT.md`), roadmap, memória de sessão (`STATE.md`) |
| [.specs/codebase/](../codebase/) | Brownfield: stack, arquitetura, convenções, testes, integrações, **CONCERNS** |
| **.specs/features/** (aqui) | Uma pasta por feature em desenvolvimento ou já especificada |

Antes de escrever `spec.md`, carregue quando existirem: `PROJECT.md`, `STATE.md`, `CONCERNS.md` e docs relevantes em `codebase/`.

## Fluxo humano (checkpoints)

1. **Checkpoint 1** — story aprovada (`story-writer`); requisitos de produto **não** são reescritos na spec.
2. **Fase 3** — `spec-writer` grava os arquivos nesta pasta.
3. **Checkpoint 2** — humano revisa os **arquivos no disco** (paths + conteúdo) e digita **APPROVE**.
4. **Fases 4–5** — builders leem `spec_paths.spec` (e `design.md` / `tasks.md` se existirem).
5. **Validator / test-verifier** — comparam código vs story + artefatos aqui.

Enquanto **Status: Awaiting human approval** no cabeçalho da spec, nenhum código de implementação da feature deve ser alterado pelo pipeline.

Após aprovação, atualize o status para **Approved** (manual ou na revisão seguinte do `spec-writer`).

## Features existentes (exemplos)

| Slug | Artefatos |
|------|-----------|
| `lazy-llm-init` | `spec.md`, `tasks.md` |
| `retry-llm-backoff` | `spec.md` |
| `portuguese-docstrings` | `spec.md` |

Use estas pastas como referência de formato; novas features devem seguir o mesmo padrão de traceability e “Files That Will Change”.

## Regras rápidas

- **Fonte de verdade de produto:** story aprovada no Checkpoint 1 — a spec **traduz**, não reinventa critérios de aceite.
- **Fonte de verdade técnica:** `spec.md` aprovado no Checkpoint 2 — nenhum arquivo de produção fora da tabela “Files That Will Change”.
- **Quick Mode** (feature trivial, ≤3 arquivos, uma frase): pula story + spec; não cria pasta aqui — só research + implementação direta.
- **Commits:** inclua mudanças em `.specs/features/<slug>/` no mesmo PR da implementação quando a spec evoluir com o código.
- **Não** rode TLC Execute neste repositório para entregar features; use o feature-factory até a Fase 8.

## Criar uma feature manualmente

Se não estiver usando o agent pipeline:

1. Crie `.specs/features/<seu-slug>/`.
2. Copie a estrutura de `spec.md` de `.claude/agents/spec-writer.md`.
3. Para features grandes, adicione `design.md` e `tasks.md` conforme `tlc-spec-driven`.
4. Abra PR com a spec **antes** ou **junto** do código — reviewers validam Checkpoint 2 por conta própria.

## Links úteis

- Orquestração: [feature-factory/SKILL.md](../../.claude/skills/feature-factory/SKILL.md)
- Autor da spec: [spec-writer.md](../../.claude/agents/spec-writer.md)
- Regras do projeto: [CLAUDE.md](../../CLAUDE.md)
