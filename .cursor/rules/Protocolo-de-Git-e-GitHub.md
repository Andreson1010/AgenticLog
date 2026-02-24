---
description: Regras para Versionamento e Git
globs: *
---

# Protocolo de Git e GitHub

1. **Branches**: Sempre sugira criar uma branch `feature/` ou `fix/` antes de propor alterações em múltiplos arquivos.
2. **Commits**: Utilize o padrão Conventional Commits (ex: `feat:`, `fix:`, `docs:`, `refactor:`).
3. **Análise de PR**: Ao analisar um Pull Request, verifique especificamente:
    - Se o bloco `try/except` no `agent.py` captura erros de rede na busca web.
    - Se as novas dependências foram adicionadas ao `requirements.txt`.
4. **Sincronização**: Lembre o usuário de dar `git pull origin main` após um merge bem-sucedido.