#!/usr/bin/env python3
"""
数据格式互转工具 — JSON / CSV / XLSX

依赖:
- Python 3.8+
- pandas
- openpyxl

安装依赖:
    pip install pandas openpyxl

功能:
- 在 JSON、CSV、XLSX 三种格式之间互相转换
- 支持命令行参数（模式、输入/输出文件、编码等）
- 支持交互式选择（无参数时进入交互模式）
- 对非表格 JSON 有降级处理（整体写入单元格）
- 可配置 JSON orient（records/columns/index/values/table）

用法示例:
    # 基本：自动从扩展名识别并转换
    python data_convert.py -i data.json -o data.csv

    # 指定模式
    python data_convert.py -m json2xlsx -i data.json -o out.xlsx --sheet Data

    # 交互式（如果未提供输入/输出路径，会进入交互选择）
    python data_convert.py

    # 将 xlsx 转为 json（以 records 形式，pretty-print）
    python data_convert.py -m xlsx2json -i book.xlsx -o out.json --json-orient records --pretty

命令行帮助:
    python data_convert.py -h

注意:
- JSON 要转换为表格时，推荐使用 list-of-dicts 或 dict-of-lists 结构。
- 对于无法解析为表格的 JSON，脚本会把整个 JSON 串放到单元格 `json` 中保存；反向转换会将其解析回原始 JSON 字符串（如果可能）。

"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd


VERSION = "0.3.0"


# --- Helpers ---

SUPPORTED_EXT = {".json", ".csv", ".xlsx", ".xls"}


def detect_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(f"不支持的文件扩展名: {ext}")
    if ext == ".json":
        return "json"
    if ext == ".csv":
        return "csv"
    return "xlsx"


def read_json_any(path: Path) -> Any:
    """支持标准 JSON 与 NDJSON (newline-delimited JSON)"""
    text = path.read_text(encoding="utf-8")
    text_stripped = text.lstrip()
    try:
        return json.loads(text)
    except Exception:
        # try NDJSON
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            raise
        try:
            objs = [json.loads(ln) for ln in lines]
            return objs
        except Exception as exc:
            raise ValueError(f"无法解析 JSON/NDJSON: {exc}") from exc


def json_to_dataframe(obj: Any) -> pd.DataFrame:
    """把各种 JSON 结构智能转换为 DataFrame。

    - list[dict] -> 常规表格
    - dict[str, list] -> 列式表格
    - 其他 -> 单列 'json' 保存原始对象的 JSON 串
    """
    if isinstance(obj, list):
        if all(isinstance(i, dict) for i in obj):
            return pd.DataFrame(obj)
        # list of scalars -> single column
        return pd.DataFrame({"value": obj})
    if isinstance(obj, dict):
        # dict of lists -> columns
        if all(isinstance(v, (list, tuple)) for v in obj.values()):
            return pd.DataFrame(obj)
        # dict of scalars -> one row
        return pd.DataFrame([obj])
    # fallback: store as single JSON string
    return pd.DataFrame({"json": [json.dumps(obj, ensure_ascii=False)]})


def dataframe_to_json_like(df: pd.DataFrame, orient: str = "records") -> Any:
    """将 DataFrame 转为 JSON-serializable Python 对象（优先 records）"""
    if orient == "records":
        return df.where(pd.notnull(df), None).to_dict(orient="records")
    if orient == "columns":
        return df.where(pd.notnull(df), None).to_dict(orient="list")
    if orient == "index":
        return df.where(pd.notnull(df), None).to_dict(orient="index")
    if orient == "values":
        return df.where(pd.notnull(df), None).values.tolist()
    if orient == "table":
        return json.loads(df.to_json(orient="table", force_ascii=False))
    raise ValueError(f"不支持的 orient: {orient}")


# --- Core conversions ---


def to_dataframe(path: Path, fmt: str, sheet: Optional[str] = None, encoding: str = "utf-8") -> pd.DataFrame:
    if fmt == "json":
        obj = read_json_any(path)
        df = json_to_dataframe(obj)
        return df
    if fmt == "csv":
        return pd.read_csv(path, dtype=object, encoding=encoding, engine="python")
    if fmt == "xlsx":
        # 默认读取第一个 sheet（或指定 sheet）
        return pd.read_excel(path, sheet_name=sheet or 0, dtype=object, engine="openpyxl")
    raise ValueError("未知输入格式")


def write_from_dataframe(df: pd.DataFrame, path: Path, fmt: str, json_orient: str = "records", pretty: bool = False, sheet: str = "Sheet1", encoding: str = "utf-8") -> None:
    if fmt == "csv":
        df.to_csv(path, index=False, encoding=encoding)
        return
    if fmt == "xlsx":
        df.to_excel(path, index=False, sheet_name=sheet or "Sheet1", engine="openpyxl")
        return
    if fmt == "json":
        obj = dataframe_to_json_like(df, orient=json_orient)
        if pretty:
            path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding=encoding)
        else:
            path.write_text(json.dumps(obj, ensure_ascii=False), encoding=encoding)
        return
    raise ValueError("未知输出格式")


# --- CLI / Interactive ---


def list_local_files(extensions: Optional[set[str]] = None) -> list[Path]:
    p = Path.cwd()
    exts = {".json", ".csv", ".xlsx", ".xls"} if extensions is None else {e.lower() for e in extensions}
    return sorted([f for f in p.iterdir() if f.suffix.lower() in exts])


def interactive_select() -> tuple[Path, str, Path, str]:
    print("交互模式 — 请选择要转换的文件（输入序号或手动输入路径）:\n")
    files = list_local_files()
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f.name}")
    raw = input("输入序号或路径: ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(files):
        in_path = files[int(raw) - 1]
    else:
        in_path = Path(raw).expanduser()
    if not in_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {in_path}")
    in_fmt = detect_format(in_path)

    print("\n请选择目标格式或直接输入输出路径:  (json / csv / xlsx)")
    out_fmt = input(f"目标格式 [json/csv/xlsx] (回车使用原扩展名转换到其它格式): ").strip().lower()
    if out_fmt not in {"json", "csv", "xlsx"}:
        # 尝试基于文件名自动推断
        default_out = in_path.with_suffix('.' + ("csv" if in_fmt != "csv" else "json"))
        out_path_raw = input(f"输入输出文件路径（回车使用 {default_out}）: ").strip()
        out_path = Path(out_path_raw).expanduser() if out_path_raw else default_out
        out_fmt = detect_format(out_path)
    else:
        out_path_raw = input(f"输出文件路径 (回车使用 ./out.{out_fmt}): ").strip()
        out_path = Path(out_path_raw).expanduser() if out_path_raw else Path.cwd() / f"out.{out_fmt}"
    return in_path, in_fmt, out_path, out_fmt


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JSON/CSV/XLSX 互转工具 — 支持交互与命令行模式")
    p.add_argument("-V", "--version", action="version", version=VERSION)
    p.add_argument("-m", "--mode", help="转换模式: json2csv/csv2json/json2xlsx/xlsx2json/csv2xlsx/xlsx2csv/auto", default="auto")
    p.add_argument("-i", "--input", help="输入文件路径")
    p.add_argument("-o", "--output", help="输出文件路径")
    p.add_argument("--json-orient", help="写出 JSON 的 orient（records/columns/index/values/table）", default="records")
    p.add_argument("--sheet", help="读取/写入的 Excel sheet 名称（可选）", default=None)
    p.add_argument("--pretty", help="美化 JSON 输出 (indent=2)", action="store_true")
    p.add_argument("--encoding", help="读写编码 (仅对 CSV/JSON 生效)", default="utf-8")
    p.add_argument("--no-interactive", help="禁止交互（若缺少参数则报错）", action="store_true")
    p.add_argument("--verbose", "-v", help="更多输出", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # 决定输入输出
    if not args.input or not args.output:
        if args.no_interactive:
            print("错误：缺少 input/output，且禁止交互。", file=sys.stderr)
            return 2
        in_path, in_fmt, out_path, out_fmt = interactive_select()
    else:
        in_path = Path(args.input).expanduser()
        out_path = Path(args.output).expanduser()
        if not in_path.exists():
            print(f"错误：找不到输入文件 {in_path}", file=sys.stderr)
            return 3
        in_fmt = detect_format(in_path)
        out_fmt = detect_format(out_path)

    # 如果用户显式提供了 mode，则以 mode 为准（但仍允许 auto）
    mode = args.mode.lower()
    if mode != "auto":
        # validate explicit mode
        if mode not in {
            "json2csv",
            "csv2json",
            "json2xlsx",
            "xlsx2json",
            "csv2xlsx",
            "xlsx2csv",
        }:
            print(f"未知模式: {mode}", file=sys.stderr)
            return 4
        # override inferred formats
        parts = mode.split("2")
        in_fmt = parts[0].replace("xls", "xlsx")
        out_fmt = parts[1].replace("xls", "xlsx")

    if args.verbose:
        print(f"输入: {in_path}  ({in_fmt}) -> 输出: {out_path}  ({out_fmt})")

    try:
        df = to_dataframe(in_path, in_fmt, sheet=args.sheet, encoding=args.encoding)
    except Exception as exc:
        print(f"读取失败: {exc}", file=sys.stderr)
        return 5

    # 写出
    try:
        write_from_dataframe(df, out_path, out_fmt, json_orient=args.json_orient, pretty=args.pretty, sheet=args.sheet or "Sheet1", encoding=args.encoding)
    except Exception as exc:
        print(f"写出失败: {exc}", file=sys.stderr)
        return 6

    print(f"已完成: {in_path.name} -> {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
