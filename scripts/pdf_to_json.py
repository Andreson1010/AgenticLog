"""Converte arquivos PDF em JSON compatível com o pipeline RAG do AgenticLog.

Formato de saída: {"PÁGINA_1": "texto...", "PÁGINA_2": "texto...", ...}

Uso:
    python scripts/pdf_to_json.py arquivo.pdf [arquivo2.pdf ...]
    python scripts/pdf_to_json.py *.pdf --output data/documents/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agenticlog.ingestion.extraction import extrair_texto_pdf
from agenticlog.shared.errors import RAGSecurityError


def pdf_para_dict(pdf_path: Path) -> dict[str, str]:
    """Extrai texto por página de um PDF (wrapper fino sobre agenticlog.rag.extrair_texto_pdf).

    Entrada: pdf_path — Path para o arquivo PDF.
    Saída: dict {"PÁGINA_1": "...", ...}.
    Lança ValueError se o PDF não contém texto extraível ou for inválido/protegido —
    preserva o contrato de exceção original do script CLI.
    """
    try:
        return extrair_texto_pdf(pdf_path)
    except RAGSecurityError as exc:
        raise ValueError(str(exc)) from exc


def converter(pdf_path: Path, output_dir: Path) -> Path:
    dados = pdf_para_dict(pdf_path)
    destino = output_dir / pdf_path.with_suffix(".json").name
    destino.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
    return destino


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converte PDFs para JSON (formato RAG AgenticLog)."
    )
    parser.add_argument("pdfs", nargs="+", type=Path, help="Arquivos PDF a converter.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/documents"),
        help="Diretório de saída (padrão: data/documents/).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    erros: list[str] = []
    for pdf in args.pdfs:
        if not pdf.exists():
            erros.append(f"Arquivo não encontrado: {pdf}")
            continue
        if pdf.suffix.lower() != ".pdf":
            erros.append(f"Não é PDF: {pdf}")
            continue
        try:
            destino = converter(pdf, args.output)
            print(f"OK  {pdf.name} -> {destino}")
        except Exception as exc:
            erros.append(f"ERRO {pdf.name}: {exc}")

    if erros:
        for e in erros:
            print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
