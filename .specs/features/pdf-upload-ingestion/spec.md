# Upload e Ingestão de PDF — Technical Spec

**Path:** `.specs/features/pdf-upload-ingestion/spec.md`
**TLC scope:** medium
**Based on story:** Como operador de logística, quero fazer upload de arquivos PDF diretamente pela interface Streamlit para indexação no ChromaDB, para que eu possa consultar contratos, relatórios e notas fiscais sem precisar converter manualmente para JSON.
**Status:** Awaiting human approval

---

## Problem Statement

Operadores de logística trabalham com contratos, relatórios e notas fiscais em formato PDF. Hoje o sistema só aceita JSON — qualquer documento PDF precisa ser convertido manualmente antes de ser ingerido. Este recurso elimina essa fricção ao permitir upload direto de PDF pela sidebar do Streamlit, com extração de texto via PyMuPDF, usando os mesmos guardrails de segurança e o mesmo fluxo de reconstrução do vectordb já existentes para JSON.

## Goals

- [ ] Operador faz upload de PDF válido pela sidebar e o texto extraído é indexado no ChromaDB sem etapas manuais.
- [ ] Todos os guardrails de segurança existentes (tamanho, nome de arquivo, duplicata) são aplicados a PDFs antes de qualquer escrita em disco.
- [ ] PDFs protegidos por senha e PDFs somente-imagem são rejeitados com mensagem clara antes de qualquer escrita em disco.
- [ ] PDFs com mix de texto e imagem são aceitos; apenas o texto extraível é indexado.
- [ ] Falha na reconstrução do vectordb após salvar o arquivo aciona rollback idêntico ao do fluxo JSON.

## Out of Scope

| Feature | Reason |
|---------|--------|
| OCR em PDFs somente-imagem | Aprovado na história: PDFs sem texto são rejeitados (AC4) |
| Ingestão incremental (sem reconstrução completa) | Aguarda feature `incremental-chroma-ingestion` (decisão tomada) |
| Preview do PDF na UI | Não consta na história aprovada |
| Edição ou exclusão de documentos pela UI | Não consta na história aprovada |
| Suporte a outros formatos (DOCX, CSV, TXT) | Não consta na história aprovada |
| Upload em lote (múltiplos PDFs de uma vez) | Não consta na história aprovada |
| Limite distinto de contagem para PDFs | Não consta na história aprovada; limite `MAX_JSON_FILES` é compartilhado |

---

## User Stories

### P1: Upload de PDF e Reconstrução do VectorDB ⭐ MVP

**User Story**: Como operador de logística, quero fazer upload de um arquivo PDF pela sidebar do Streamlit, para que o texto do documento seja indexado no ChromaDB e eu possa consultá-lo na próxima query.

**Why P1**: Capacidade central e única da história. Sem ela a feature não existe.

**Acceptance Criteria**:

1. QUANDO operador seleciona PDF válido e clica "Ingerir Documento" ENTÃO sistema SHALL extrair texto via PyMuPDF, salvar arquivo em `DIR_DOCUMENTS`, reconstruir vectordb com `reconstruir_vectordb()`, exibir `st.success` e chamar `st.rerun()`. *(AC1)*
2. QUANDO PDF ingerido com sucesso ENTÃO agente SHALL recuperar chunks no step `retrieve` em queries subsequentes. *(AC2)*
3. QUANDO PDF está protegido por senha ENTÃO sistema SHALL lançar `RAGSecurityError` com mensagem portuguesa e não salvar o arquivo. *(AC3)*
4. QUANDO PDF é somente-imagem (zero caracteres não-brancos extraíveis) ENTÃO sistema SHALL lançar `RAGSecurityError` e não salvar o arquivo. *(AC4)*
5. QUANDO PDF excede `MAX_JSON_FILE_SIZE_MB` (10 MB) ENTÃO sistema SHALL lançar `RAGSecurityError` antes de qualquer extração de texto. *(AC5)*
6. QUANDO nome do arquivo já existe em `DIR_DOCUMENTS` ENTÃO sistema SHALL lançar `RAGSecurityError("Arquivo com esse nome já existe.")`. *(AC6)*
7. QUANDO nome do arquivo contém `INVALID_FILENAME_CHARS` ou é um nome reservado Windows ENTÃO `_sanitizar_nome_arquivo` SHALL lançar `RAGSecurityError`. *(AC7)*
8. QUANDO `reconstruir_vectordb()` falha após arquivo salvo ENTÃO sistema SHALL chamar `unlink` no arquivo salvo e exibir `st.error` (rollback idêntico ao JSON). *(AC8)*
9. QUANDO arquivo enviado tem extensão diferente de `.pdf` (case-insensitive) ENTÃO sistema SHALL lançar `RAGSecurityError("Apenas arquivos .pdf são aceitos.")`. *(AC9)*
10. QUANDO PDF contém mix de texto e imagens ENTÃO sistema SHALL aceitar e indexar apenas o texto extraível. *(AC11+AC12)*

**Independent Test**: Upload de PDF com texto válido → arquivo em `DIR_DOCUMENTS` → vectordb reconstruído → query retorna chunks do PDF; upload de PDF com senha → zero bytes em disco → `st.error` exibido.

---

## Edge Cases

- QUANDO extensão é `.PDF` (maiúsculo) ENTÃO sistema SHALL aceitar (verificação case-insensitive via `Path(filename).suffix.lower()`). *(AC10)*
- QUANDO PDF tem algumas páginas com texto e outras somente-imagem ENTÃO sistema SHALL aceitar o arquivo e indexar apenas o texto das páginas com texto (AC12).
- QUANDO `fitz.open()` levanta `fitz.FileDataError` (arquivo corrompido ou senha) ENTÃO `extrair_texto_pdf` SHALL lançar `RAGSecurityError`.
- QUANDO `doc.needs_pass == True` (detectado antes de tentativa de leitura) ENTÃO `extrair_texto_pdf` SHALL lançar `RAGSecurityError("PDF protegido por senha.")`.
- QUANDO texto extraído contém apenas espaços/tabs/newlines ENTÃO sistema SHALL rejeitar como PDF somente-imagem (zero texto real).
- QUANDO `cria_vectordb()` é chamado com `DIR_DOCUMENTS` contendo tanto `.json` quanto `.pdf`, ambos os formatos SHALL ser carregados (requer extensão do `glob` atual).

---

## Requirement Traceability

| Requirement ID | Critério de Aceite | Phase | Status |
|----------------|--------------------|-------|--------|
| PDF-01 | AC1 — happy path: extrair + salvar + reconstruir + success + rerun | Design | Pending |
| PDF-02 | AC2 — agente recupera chunks de PDF após ingestão | Design | Pending |
| PDF-03 | AC3 — PDF com senha → RAGSecurityError, sem escrita | Design | Pending |
| PDF-04 | AC4 — PDF somente-imagem → RAGSecurityError, sem escrita | Design | Pending |
| PDF-05 | AC5 — PDF > 10 MB → RAGSecurityError antes de extração | Design | Pending |
| PDF-06 | AC6 — nome duplicado → RAGSecurityError | Design | Pending |
| PDF-07 | AC7 — nome inválido/reservado → RAGSecurityError via `_sanitizar_nome_arquivo` | Design | Pending |
| PDF-08 | AC8 — rollback: unlink + st.error se vectordb falha | Design | Pending |
| PDF-09 | AC9 — extensão não-.pdf → RAGSecurityError | Design | Pending |
| PDF-10 | AC10 — extensão case-insensitive (.PDF aceito) | Design | Pending |
| PDF-11 | AC11+AC12 — mix texto+imagem: aceitar, indexar só texto | Design | Pending |

---

## Data Model Changes

Nenhuma migração de schema. ChromaDB é reconstruído do zero a cada chamada a `cria_vectordb()`.

Nenhuma constante nova em `config.py` é necessária. As constantes reutilizadas são:
- `MAX_JSON_FILE_SIZE_MB = 10` — limite de tamanho compartilhado com JSON.
- `MAX_JSON_FILES = 1000` — limite de contagem compartilhado.
- `INVALID_FILENAME_CHARS`, `WINDOWS_RESERVED_NAMES` — reutilizados sem alteração.
- `DIR_DOCUMENTS` — mesmo diretório de destino dos JSONs.

---

## Novas Funções — `rag.py`

### `extrair_texto_pdf(path: Path) -> str`

```python
def extrair_texto_pdf(path: Path) -> str:
    """Extrai texto de um arquivo PDF usando PyMuPDF (fitz).

    Entrada: path — Path para um arquivo PDF temporário já salvo em disco.
    Saída: string com todo o texto extraível (páginas concatenadas).
    Lança RAGSecurityError se:
      - PDF está protegido por senha (doc.needs_pass == True).
      - fitz.open() lança fitz.FileDataError (arquivo corrompido).
      - Texto extraído tem zero caracteres não-brancos (PDF somente-imagem).
    """
```

Comportamento detalhado:
1. `import fitz; doc = fitz.open(str(path))` — captura `fitz.FileDataError` → `RAGSecurityError`.
2. `if doc.needs_pass` → `RAGSecurityError("PDF protegido por senha.")`.
3. Concatena `page.get_text()` para todas as páginas.
4. `if not texto.strip()` → `RAGSecurityError("PDF não contém texto extraível (somente imagem).")`.
5. Retorna texto completo.

### `salvar_pdf_enviado(filename: str, conteudo: bytes) -> Path`

```python
def salvar_pdf_enviado(filename: str, conteudo: bytes) -> Path:
    """Valida e persiste um arquivo PDF enviado pelo operador.

    Entrada:
      filename — nome original do arquivo (str).
      conteudo — conteúdo binário do arquivo (bytes).
    Saída: Path do arquivo salvo em DIR_DOCUMENTS.
    Lança RAGSecurityError em qualquer falha de validação.
    """
```

Ordem obrigatória das validações (espelha `salvar_documento_enviado`):
1. **Extensão** — `Path(filename).suffix.lower() != ".pdf"` → `RAGSecurityError("Apenas arquivos .pdf são aceitos.")`.
2. **Tamanho** — `len(conteudo) > MAX_JSON_FILE_SIZE_MB * 1024 * 1024` → `RAGSecurityError`.
3. **Nome** — `_sanitizar_nome_arquivo(filename)` (função existente, sem alteração).
4. **Duplicata** — `(DIR_DOCUMENTS / safe_name).exists()` → `RAGSecurityError("Arquivo com esse nome já existe.")`.
5. **Contagem** — `len(list(DIR_DOCUMENTS.glob("*.pdf"))) + len(list(DIR_DOCUMENTS.glob("*.json"))) + 1 > MAX_JSON_FILES` → `RAGSecurityError`.
6. **Conteúdo** — escreve em `tempfile`, chama `extrair_texto_pdf(tmp_path)`; se lançar `RAGSecurityError`, apaga tempfile e relança.
7. `shutil.move(str(tmp_path), DIR_DOCUMENTS / safe_name)` — move arquivo apenas após todas as validações passarem.

### Alteração em `cria_vectordb()` — `rag.py`

A linha atual com `DirectoryLoader` para JSON permanece inalterada. PDFs são carregados via `langchain_core.documents.Document` construídos manualmente, sem `PyMuPDFLoader`:

```python
from langchain_core.documents import Document

# JSON (inalterado)
json_docs = DirectoryLoader(str(DIR_DOCUMENTS), glob="*.json", ...).load()

# PDF — sem dependência de PyMuPDFLoader
pdf_docs = []
for pdf_path in DIR_DOCUMENTS.glob("*.pdf"):
    texto = extrair_texto_pdf(pdf_path)
    pdf_docs.append(Document(page_content=texto, metadata={"source": str(pdf_path)}))

documents = json_docs + pdf_docs
text_splitter.split_documents(documents)  # inalterado
```

Justificativa: `langchain_core.documents.Document` é dependência core — sempre disponível, sem risco de versão. Reutiliza `extrair_texto_pdf` (validação já feita) sem duplicação. Evita dependência de `PyMuPDFLoader` do `langchain-community`, que pode não estar disponível em todas as versões instaladas.

---

## Process / Background Flow

**Happy path:**
1. Operador abre sidebar expander "Adicionar Documento".
2. Operador seleciona arquivo `.pdf` via `st.file_uploader`.
3. Operador clica "Ingerir Documento".
4. `app.py` detecta extensão `.pdf` (case-insensitive) e chama `salvar_pdf_enviado(filename, conteudo)`.
5. `salvar_pdf_enviado` executa pipeline de validação completo; escreve arquivo em `DIR_DOCUMENTS`.
6. `app.py` exibe spinner "Reconstruindo base vetorial...".
7. `reconstruir_vectordb()` → `cria_vectordb()` carrega JSONs + PDFs → gera embeddings → persiste ChromaDB.
8. `st.success("Documento ingerido com sucesso.")` → `st.rerun()`.

**Failure path — erro de validação:**
1. Passos 1–4 iguais.
2. `salvar_pdf_enviado` lança `RAGSecurityError`.
3. `app.py` captura, chama `st.error(str(e))`.
4. Zero bytes escritos em disco; sem rebuild.

**Failure path — falha no rebuild:**
1. Passos 1–7 iguais; arquivo está em disco.
2. `reconstruir_vectordb()` lança qualquer exceção.
3. `app.py` chama `saved_path.unlink(missing_ok=True)` e `st.error(...)`.
4. `data/vectordb/` original permanece intacto.

---

## API Changes

Nenhuma alteração de API HTTP. Toda interação é via UI Streamlit.

---

## Frontend Changes

**`app.py` — sidebar (expander "Adicionar Documento" já existente, linha 80):**

Alterações necessárias:
1. `st.file_uploader` — adicionar `"pdf"` ao parâmetro `type` (ou manter `type=None` e validar manualmente — preferir `type=None` para consistência com JSON).
2. Atualizar label: `"Selecione um arquivo JSON ou PDF"`.
3. Em `_ingerir_documento`: adicionar branch por extensão:
   - `suffix == ".pdf"` → chama `salvar_pdf_enviado(filename, conteudo)`.
   - caso contrário → chama `salvar_documento_enviado(filename, conteudo)` (comportamento atual).
4. Lógica pós-salvamento (spinner + `reconstruir_vectordb` + `st.success` + `st.rerun` + rollback) permanece **idêntica** para ambos os formatos.

Exemplo de estrutura do branch em `_ingerir_documento`:

```python
suffix = Path(filename).suffix.lower()
if suffix == ".pdf":
    saved_path = salvar_pdf_enviado(filename, conteudo)
else:
    saved_path = salvar_documento_enviado(filename, conteudo)
```

---

## Tests Required

### Unitários — `tests/test_rag.py`

Adicionar classes `TestSalvarPdfEnviado` e `TestExtrairTextoPdf`.

| Método | AC coberto | O que testa |
|--------|-----------|-------------|
| `teste_1_salvar_pdf_sucesso` | AC1 | PDF válido → arquivo salvo em DIR_DOCUMENTS |
| `teste_2_salvar_rejeita_extensao_invalida` | AC9 | `.txt` → RAGSecurityError |
| `teste_3_salvar_rejeita_extensao_pdf_maiusculo_aceito` | AC10 | `.PDF` → aceito (case-insensitive) |
| `teste_4_salvar_rejeita_tamanho_excedido` | AC5 | > 10 MB → RAGSecurityError antes de extração |
| `teste_5_salvar_rejeita_colisao_de_nome` | AC6 | nome duplicado → RAGSecurityError |
| `teste_6_salvar_rejeita_path_traversal` | AC7 | `../evil.pdf` → RAGSecurityError |
| `teste_7_salvar_rejeita_nome_reservado_windows` | AC7 | `CON.pdf` → RAGSecurityError |
| `teste_8_extrair_texto_pdf_sucesso` | AC1 | PDF com texto → retorna string não-vazia |
| `teste_9_extrair_rejeita_pdf_com_senha` | AC3 | `doc.needs_pass == True` → RAGSecurityError |
| `teste_10_extrair_rejeita_pdf_somente_imagem` | AC4 | texto em branco → RAGSecurityError |
| `teste_11_extrair_aceita_mix_texto_imagem` | AC11+AC12 | texto parcial → retorna texto, sem erro |
| `teste_12_salvar_rejeita_pdf_corrompido` | AC3 | `fitz.FileDataError` → RAGSecurityError |

Padrão de mock para `fitz`:
```python
@patch("agenticlog.rag.fitz.open")
def teste_9_extrair_rejeita_pdf_com_senha(self, mock_fitz_open):
    mock_doc = MagicMock()
    mock_doc.needs_pass = True
    mock_fitz_open.return_value = mock_doc
    with self.assertRaises(RAGSecurityError):
        extrair_texto_pdf(Path("qualquer.pdf"))
```

### Testes de Aceitação — `tests/acceptance/test_document_ingestion_ui.py`

Adicionar nova classe `TestPDFIngestion` (não alterar classes DOCING existentes). Atualizar docstring do módulo para referenciar os novos IDs PDF-01 a PDF-11.

| Método | Req ID | O que testa |
|--------|--------|-------------|
| `test_pdf_01_happy_path` | PDF-01 | Fluxo completo mock: `salvar_pdf_enviado` + `reconstruir_vectordb` + `st.success` + `st.rerun` |
| `test_pdf_03_senha_mostra_erro` | PDF-03 | `salvar_pdf_enviado` lança `RAGSecurityError` → `st.error`, sem `rerun` |
| `test_pdf_04_somente_imagem_mostra_erro` | PDF-04 | Idem PDF-03 com mensagem "somente imagem" |
| `test_pdf_05_tamanho_excedido` | PDF-05 | Arquivo > 10 MB → `st.error` |
| `test_pdf_06_nome_duplicado` | PDF-06 | `RAGSecurityError("Arquivo com esse nome já existe.")` → `st.error` |
| `test_pdf_08_rollback_rebuild_falha` | PDF-08 | `reconstruir_vectordb` lança exceção → `saved_path.unlink` chamado + `st.error` |
| `test_pdf_09_extensao_invalida` | PDF-09 | `.docx` → `st.error("Apenas arquivos .pdf são aceitos.")` |
| `test_pdf_10_extensao_maiuscula_aceita` | PDF-10 | `.PDF` → `salvar_pdf_enviado` chamado (sem erro de extensão) |

Também atualizar `TestDOCING03NonJsonExtensionRejected` — o teste `test_docing_03_salvar_rejeita_extensao_nao_json_diretamente` continuará passando pois `salvar_documento_enviado` permanece inalterado.

### Testes de cobertura

```bash
pytest --cov=agenticlog --cov-fail-under=80 -v
```

### Testes existentes que podem quebrar

| Arquivo | Risco | Mitigação |
|---------|-------|-----------|
| `tests/test_rag.py` — testes de `cria_vectordb` | Chamada ao `DirectoryLoader` agora ocorre duas vezes (JSON + PDF) | Atualizar mocks para retornar lista vazia para PDFs |
| `tests/acceptance/test_document_ingestion_ui.py` — `TestDOCING03` | `_ingerir_documento` agora tem branch por extensão | Mocks existentes cobrem apenas `.json`; branch `.pdf` não afeta testes DOCING |

---

## Files That Will Change

| Arquivo | Tipo de alteração | Por que |
|---------|-------------------|---------|
| `src/agenticlog/rag.py` | Adicionar funções; modificar `cria_vectordb` | Novas funções `extrair_texto_pdf` e `salvar_pdf_enviado`; `cria_vectordb` precisa carregar PDFs além de JSONs |
| `app.py` | Modificar `_ingerir_documento`; atualizar label do `file_uploader` | Branch por extensão para rotear para `salvar_pdf_enviado` vs `salvar_documento_enviado` |
| `tests/test_rag.py` | Adicionar casos de teste | 12 novos métodos em duas novas classes de teste |
| `tests/acceptance/test_document_ingestion_ui.py` | Adicionar classe + atualizar docstring | 8 novos métodos de aceitação para fluxo PDF |
| `src/agenticlog/config.py` | Sem alteração | Todas as constantes necessárias já existem |

Linhas de referência em `rag.py`:
- Linha 160: `salvar_documento_enviado` — template para `salvar_pdf_enviado` (não modificar).
- Linha 169: verificação de extensão `.json` — NÃO alterar; nova função usa `.pdf`.
- Linha 212: `cria_vectordb()` — adicionar segundo `DirectoryLoader` após linha 238.

Linhas de referência em `app.py`:
- Linha 15: import de `salvar_documento_enviado` — adicionar `salvar_pdf_enviado`.
- Linha 17: `_ingerir_documento` — adicionar branch por extensão.
- Linha 81: `st.file_uploader` label — atualizar.

---

## Risks

| Risco | Severidade | Mitigação |
|-------|-----------|-----------|
| `cria_vectordb()` usa glob `"*.json"` — PDFs em `DIR_DOCUMENTS` são ignorados sem a alteração | ALTA | Especificado explicitamente; testes de unidade de `cria_vectordb` devem cobrir o segundo loader |
| PDFs com senha: `fitz.open()` pode não levantar exceção imediatamente em todos os backends do PyMuPDF — `doc.needs_pass` deve ser verificado explicitamente | ALTA | `extrair_texto_pdf` verifica `doc.needs_pass` antes de ler páginas |
| PDFs somente-imagem: `fitz.open()` tem sucesso mas `page.get_text()` retorna string vazia — sem checagem explícita o documento seria indexado como string vazia | ALTA | `extrair_texto_pdf` verifica `texto.strip()` após extração |
| Conflito de sequenciamento: `incremental-chroma-ingestion` planeja substituir `reconstruir_vectordb` — esta feature usa `reconstruir_vectordb` diretamente | MEDIA | Decisão tomada e documentada; quando `incremental-chroma-ingestion` aterrissar, ambos os formatos migram juntos |
| ~~`PyMuPDFLoader` do LangChain~~ | ~~MEDIA~~ | RESOLVIDO: usar `Document` objects manualmente via `extrair_texto_pdf` — sem dependência de `langchain-community` |
| Arquivo PDF salvo em disco antes de `reconstruir_vectordb` ser chamado — se processo for morto entre os dois passos, arquivo orphan permanece em `DIR_DOCUMENTS` | BAIXA | Na próxima reconstrução (CLI ou próximo upload bem-sucedido) todos os arquivos em `DIR_DOCUMENTS` são indexados; não causa corrupção |
| Race condition: dois uploads simultâneos interfolham verificação de contagem e escrita | BAIXA | Sem lock em escopo; guarda de contagem é best-effort (igual ao fluxo JSON) |
| `fitz.FileDataError` pode ter nome diferente dependendo da versão do PyMuPDF | BAIXA | Tratar também `Exception` genérica com mensagem de erro amigável; documentar versão mínima |

---

## Open Questions

Todas resolvidas antes da especificação (decisões do aprovador):

| Questão | Resolução |
|---------|-----------|
| Qual diretório para salvar PDFs? | `DIR_DOCUMENTS` — mesmo que JSON (RESOLVIDO) |
| Qual limite de tamanho? | `MAX_JSON_FILE_SIZE_MB = 10` compartilhado (RESOLVIDO) |
| Como reconstruir vectordb? | `reconstruir_vectordb()` — sem aguardar incremental (RESOLVIDO) |
| PDFs com mix texto+imagem: aceitar ou rejeitar? | Aceitar parcialmente, indexar só texto extraível (RESOLVIDO) |

---

## Success Criteria

- [ ] Todos os 12 testes unitários em `TestSalvarPdfEnviado` e `TestExtrairTextoPdf` passam.
- [ ] Todos os 8 testes de aceitação em `TestPDFIngestion` passam.
- [ ] Testes DOCING-01 a DOCING-10 existentes continuam passando sem alteração.
- [ ] `pytest --cov=agenticlog --cov-fail-under=80` passa.
- [ ] Smoke test manual: upload de PDF com texto → query retorna conteúdo do PDF nos documentos recuperados.
- [ ] Smoke test manual: upload de PDF com senha → zero bytes em `DIR_DOCUMENTS` → `st.error` exibido.
- [ ] Smoke test manual: upload de PDF somente-imagem → zero bytes em `DIR_DOCUMENTS` → `st.error` exibido.
- [ ] Smoke test manual: `cria_vectordb()` com pasta contendo JSONs e PDFs → ambos os formatos indexados.
