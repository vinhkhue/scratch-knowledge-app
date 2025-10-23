#!/usr/bin/env python3
"""
Extract entities và relationships từ Scratch files với LLM
- Tích hợp tất cả improvements: clean entities, improved relationships, rate limit handling
- Chỉ giữ lại file này để extract data
"""

from __future__ import annotations

import os
import json
import re
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import pandas as pd

# -------------------- logging --------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("scratch_kg")

# -------------------- OpenAI client --------------------
try:
    from openai import OpenAI  # SDK mới
    _sdk_mode = "new"
except Exception:  # pragma: no cover
    import openai as _openai
    OpenAI = None
    _sdk_mode = "old"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required in environment")

if _sdk_mode == "new":
    client = OpenAI(api_key=OPENAI_API_KEY)

    def chat_completion(model: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8 seconds
                    log.warning(f"Rate limit hit, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    log.error(f"API error: {e}")
                    raise
        raise Exception(f"Failed after {max_retries} attempts due to rate limits")
else:  # very old SDK fallback
    _openai.api_key = OPENAI_API_KEY  # type: ignore

    def chat_completion(model: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:  # pragma: no cover
        resp = _openai.ChatCompletion.create(  # type: ignore
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp["choices"][0]["message"]["content"] or ""

MODEL_EXTRACT = os.getenv("OPENAI_MODEL_EXTRACT", "gpt-4o-mini")
MODEL_REL = os.getenv("OPENAI_MODEL_REL", "gpt-4o")

# -------------------- Helpers --------------------
_md_fence = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def strip_md_fences(text: str) -> str:
    return _md_fence.sub("", text).strip()


def extract_json_payload(text: str) -> Dict[str, Any]:
    """Parse JSON từ LLM, chịu lỗi. Nếu fail thì cắt khối { ... } đầu tiên hợp lệ."""
    t = strip_md_fences(text)
    try:
        return json.loads(t)
    except Exception:
        pass
    start = t.find('{')
    end = t.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = t[start:end+1]
        # cân bằng ngoặc
        depth = 0
        last_ok = -1
        for i, ch in enumerate(candidate):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
            if depth == 0:
                last_ok = i
        if last_ok != -1:
            candidate = candidate[:last_ok+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    log.warning("Không parse được JSON thuần túy từ LLM. Trả về {}.")
    return {}


def safe_list(d: Dict[str, Any], *keys: str) -> List[Any]:
    for k in keys:
        if isinstance(d.get(k), list):
            return d[k]  # type: ignore
    return []


def stable_id(prefix: str, text: str) -> str:
    h = hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]
    return f"{prefix}_{h}"


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().casefold()


def smart_truncate(text: str, max_chars: int = 60000) -> str:
    """Truncate at word boundary - reduced to avoid rate limits"""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > max_chars - 100:  # Within 100 chars of limit
        return truncated[:last_space]
    return truncated


# -------------------- LLM prompts tối ưu theo 2 file --------------------
# Blocks.txt có các bảng bắt đầu bằng dòng "Tên Khối - Tiếng Anh\tChức năng ..."
BLOCK_SYS = (
    "Bạn sẽ trích xuất KHỐI LỆNH Scratch 3.0 từ các BẢNG có tiêu đề 'Tên Khối - Tiếng Anh' trong văn bản. BỎ QUA mô tả chung.\n"
    "YÊU CẦU: Chuẩn hoá tên khối đúng chuẩn Scratch 3.0 (tiếng Anh), giữ nguyên dấu ngoặc tham số. Suy luận category theo mục tiêu đề (Motion, Looks, Sound, Events, Control, Sensing, Operators, Variables, Lists, My Blocks, Extensions). Suy luận shape nếu có (hat|stack|reporter|boolean|cap|c). Chấp nhận tiếng Việt và lỗi OCR.\n"
    "TRẢ JSON THUẦN (không markdown):\n"
    '{"entities": [{"name":"<name_en>","description":"<vi_description>","type":"block","category":"Motion","shape":"stack","args":["steps","degrees"],"aliases_vi":["đi bước","quay"],"importance":"high"}]}'
)

# scratch_clean.txt chứa sách Tin học 8 có lỗi OCR, chỉ lấy phần LIÊN QUAN tới Scratch
EDU_SYS = (
    "Trích xuất CÁC KHÁI NIỆM LIÊN QUAN SCRATCH từ sách 'Tin học 8' có NHIỀU LỖI OCR. BỎ các phần không liên quan Scratch. Chuẩn hoá chính tả tiếng Việt.\n"
    "ƯU TIÊN: rẽ nhánh, điều kiện, vòng lặp, biến, danh sách, sự kiện, thông điệp (broadcast), sprite, sân khấu, toạ độ, cảm biến, toán tử, nhập/xuất, song song, thuật toán. Gắn aliases nếu có.\n"
    "TRẢ JSON THUẦN:\n"
    '{"entities": [{"name":"vòng lặp","description":"Lặp lại hành động trong Scratch","type":"concept","category":"control_flow","aliases":["repeat","forever","lặp lại"],"importance":"high"}]}'
)

REL_SYS = (
    "Bạn là chuyên gia ghép KHỐI với KHÁI NIỆM Scratch từ hai nguồn.\n"
    "Chỉ dùng tên có trong danh sách cung cấp. KHÔNG tự bịa tên mới. Nếu không tìm được quan hệ hợp lệ, trả về relationships: [].\n"
    "Tạo >=50 quan hệ nếu có: block belongs_to category; block implements concept; concept demonstrated_by block; block uses/relates_to sprite/stage/biến/danh sách.\n"
    "TRẢ JSON THUẦN: {\n"
    " \"relationships\": [{\n"
    " \"source\": \"<entity nguồn>\",\n"
    " \"target\": \"<entity đích>\",\n"
    " \"relation\": \"mô tả ngắn\",\n"
    " \"type\": \"belongs_to|implements|demonstrated_by|relates_to|uses\",\n"
    " \"strength\": \"strong|medium|weak\"\n"
    " }]\n"
    "}"
)

# -------------------- LLM extractors --------------------
def llm_extract_blocks(content: str, want: int = 40) -> List[Dict[str, Any]]:
    # Chỉ giữ các đoạn bảng để giảm nhiễu
    kept: List[str] = []
    for m in re.finditer(r"(?:Tên Khối - Tiếng Anh).*?(?=\n\s*\d+\.\s|\n\s*$)", content, flags=re.IGNORECASE|re.DOTALL):
        kept.append(m.group(0))
    snippet = "\n\n".join(kept) if kept else content
    prompt = (
        f"Hãy liệt kê ~{want} KHỐI từ các bảng 'Tên Khối - Tiếng Anh' dưới đây.\n"
        f"Chỉ lấy tên khối chuẩn tiếng Anh và mô tả ngắn tiếng Việt, kèm category, shape và args nếu có.\n\n"
        f"{snippet[:2600]}"
    )
    raw = chat_completion(MODEL_EXTRACT, BLOCK_SYS, prompt, temperature=0.1, max_tokens=2500)
    data = extract_json_payload(raw)
    ents = safe_list(data, 'entities', 'Entities')
    out: List[Dict[str, Any]] = []
    for e in ents:
        name = str(e.get('name', '')).strip()
        if not name:
            continue
        out.append({
            'name': name,
            'description': e.get('description', ''),
            'type': 'block',
            'category': e.get('category', 'unknown'),
            'shape': e.get('shape', None),
            'args': e.get('args', []),
            'aliases_vi': e.get('aliases_vi', []),
            'importance': e.get('importance', 'medium'),
        })
    log.info(f"Blocks extracted: {len(out)}")
    return out


def llm_extract_edu(content: str, chunk_size: int = 1600, overlap: int = 300) -> List[Dict[str, Any]]:
    all_ents: List[Dict[str, Any]] = []
    step = max(1, chunk_size - overlap)
    key_re = re.compile(r"scratch|khối|sprite|sân khấu|biến\s|danh\s*sách|vòng\s*lặp|rẽ\s*nhánh|điều\s*kiện|toán\s*tử|sự kiện", re.IGNORECASE)
    for idx, start in enumerate(range(0, len(content), step), start=1):
        chunk = content[start:start+chunk_size]
        if not key_re.search(chunk):
            continue  # bỏ phần không liên quan Scratch
        prompt = (
            "Trích xuất 8–15 concept/liên quan Scratch. Chuẩn hoá chính tả, gom từ đồng nghĩa, thêm aliases nếu có.\n" +
            chunk[:1300]
        )
        try:
            raw = chat_completion(MODEL_EXTRACT, EDU_SYS, prompt, temperature=0.1, max_tokens=1800)
            data = extract_json_payload(raw)
            ents = safe_list(data, 'entities', 'Entities')
            for e in ents:
                name = str(e.get('name', '')).strip()
                if not name:
                    continue
                all_ents.append({
                    'name': name,
                    'description': e.get('description', ''),
                    'type': e.get('type', 'concept'),
                    'category': e.get('category', 'basic'),
                    'aliases': e.get('aliases', []),
                    'importance': e.get('importance', 'medium'),
                })
            log.info(f"Chunk {idx}: +{len(ents)} entities")
        except Exception as ex:
            log.error(f"Chunk {idx} error: {ex}")
            continue
    # de-dup by normalized name
    uniq: Dict[str, Dict[str, Any]] = {}
    for e in all_ents:
        key = normalize(e['name'])
        if key not in uniq:
            uniq[key] = e
    log.info(f"Edu entities unique: {len(uniq)}")
    return list(uniq.values())


def llm_cross_relationships(block_entities: List[Dict[str, Any]], edu_entities: List[Dict[str, Any]], blocks_ctx: str, edu_ctx: str, target: int = 80) -> List[Dict[str, Any]]:
    block_names = ", ".join([e['name'] for e in block_entities[:60]])
    edu_names = ", ".join([e['name'] for e in edu_entities[:80]])
    fewshot = (
        "Ví dụ: {\n"
        " \"relationships\": [\n"
        " {\"source\": \"repeat ()\", \"target\": \"vòng lặp\", \"relation\": \"repeat implements loop\", \"type\": \"implements\", \"strength\": \"strong\"},\n"
        " {\"source\": \"when green flag clicked\", \"target\": \"sự kiện bắt đầu chương trình\", \"relation\": \"hat block triggers scripts\", \"type\": \"implements\", \"strength\": \"strong\"}\n"
        " ]\n"
        "}"
    )
    prompt = (
        f"Ghép quan hệ từ DANH SÁCH cho trước. Không dùng tên ngoài danh sách. Ít nhất {min(50, target)} nếu văn bản cho phép.\n"
        f"Blocks: {block_names}\nConcepts: {edu_names}\n\n"
        f"Blocks Context:\n{smart_truncate(blocks_ctx, 30000)}\n\n"
        f"Edu Context:\n{smart_truncate(edu_ctx, 30000)}\n\n"
        f"{fewshot}"
    )
    raw = chat_completion(MODEL_REL, REL_SYS, prompt, temperature=0.0, max_tokens=3200)
    data = extract_json_payload(raw)
    rels = safe_list(data, 'relationships', 'Relations')
    out: List[Dict[str, Any]] = []
    for r in rels:
        s = str(r.get('source', '')).strip()
        t = str(r.get('target', '')).strip()
        if not s or not t:
            continue
        out.append({
            'source': s,
            'target': t,
            'relation': r.get('relation', ''),
            'type': r.get('type', 'relates_to'),
            'strength': r.get('strength', 'medium'),
        })
    log.info(f"Relationships extracted: {len(out)}")
    return out


# -------------------- KG build --------------------
def find_entity_id_by_name(entities_df: pd.DataFrame, name: str) -> Optional[str]:
    """Fuzzy + alias matching, optimized."""
    n = normalize(name)
    
    # Exact match
    m = entities_df[entities_df['title'].str.casefold() == n]
    if not m.empty:
        return str(m.iloc[0]['id'])
    
    # Contains match (escape special regex chars)
    try:
        escaped = re.escape(name) if any(c in name for c in r'\.^$*+?{}[]()') else name
        m = entities_df[entities_df['title'].str.contains(escaped, case=False, na=False, regex=True)]
        if not m.empty:
            return str(m.iloc[0]['id'])
    except Exception:
        pass
    
    # Alias matching in metadata (vectorized approach would be complex, keep simple)
    for _, row in entities_df.iterrows():
        try:
            meta = row['metadata']
            if not isinstance(meta, dict):
                continue
            aliases: List[str] = []
            for k in ('aliases', 'aliases_vi'):
                v = meta.get(k)
                if isinstance(v, list):
                    aliases.extend([str(x) for x in v])
            for al in aliases:
                if normalize(al) == n or (len(al) > 3 and al.lower() in name.lower()):
                    return str(row['id'])
        except Exception:
            continue
    
    # Keyword partials
    for kw in [x for x in re.split(r"\W+", name) if len(x) > 3]:
        try:
            escaped = re.escape(kw) if any(c in kw for c in r'\.^$*+?{}[]()') else kw
            m = entities_df[entities_df['title'].str.contains(escaped, case=False, na=False, regex=True)]
            if not m.empty:
                return str(m.iloc[0]['id'])
        except Exception:
            pass
    
    return None


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_two_inputs(cli_paths: Optional[List[str]] = None) -> Tuple[List[Path], Dict[str, str], str]:
    files: List[Path] = []
    sources: Dict[str, str] = {}
    if cli_paths:
        for p in cli_paths:
            path = Path(p)
            if path.exists():
                files.append(path)
    if not files:
        for p in ["/mnt/data/Blocks.txt", "/mnt/data/scratch_clean.txt"]:
            path = Path(p)
            if path.exists():
                files.append(path)
    if not files:
        for path in Path("input").glob("*.txt"):
            files.append(path)
    if not files:
        raise FileNotFoundError("Không tìm thấy file .txt nào. Hãy cung cấp đường dẫn hoặc đặt vào /mnt/data hoặc input/")
    all_content: List[str] = []
    for f in files:
        if "Blocks" in f.name:
            src = "Scratch Wiki"
        elif "tin-hoc-8" in f.name or "scratch_clean" in f.name:
            src = "Tin học 8 - Cánh Diều"
        else:
            src = "Unknown"
        sources[f.name] = src
        txt = f.read_text(encoding='utf-8', errors='ignore')
        all_content.append(f"\n\n=== {f.name} ({src}) ===\n\n{txt}")
        log.info(f"Read {f.name} ({src}) len={len(txt)})")
    return files, sources, "\n".join(all_content)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception as ex:
        log.warning(f"to_parquet lỗi: {ex}. Ghi CSV fallback: {path.with_suffix('.csv')}")
        df.to_csv(path.with_suffix('.csv'), index=False)


def build_kg(input_paths: Optional[List[str]] = None) -> None:
    out_dir = Path("output")
    ensure_output_dir(out_dir)
    files, file_sources, joined_content = read_two_inputs(input_paths)
    
    # LLM extracts
    block_entities: List[Dict[str, Any]] = []
    edu_entities: List[Dict[str, Any]] = []
    blocks_ctx = ""
    edu_ctx = ""
    for f in files:
        txt = Path(f).read_text(encoding='utf-8', errors='ignore')
        if "Blocks" in f.name:
            log.info(f"Extract blocks from {f.name}")
            blocks_ctx = txt
            block_entities.extend(llm_extract_blocks(txt))
        else:
            log.info(f"Extract edu concepts from {f.name}")
            edu_ctx += ("\n" + txt)
            edu_entities.extend(llm_extract_edu(txt))
    
    # Cross relationships
    rels = llm_cross_relationships(block_entities, edu_entities, blocks_ctx, edu_ctx)
    
    # documents.parquet
    documents_df = pd.DataFrame({
        'id': ['doc_1'],
        'text': [joined_content],
        'title': ['Scratch Knowledge Base - Tin học 8 + Scratch Wiki'],
        'creation_date': ['2024-01-01T00:00:00Z'],
        'metadata': [{
            'source': 'mixed',
            'subject': 'scratch',
            'files': list(file_sources.keys()),
            'sources': list(file_sources.values()),
        }],
    })
    write_parquet(documents_df, out_dir / 'documents.parquet')
    
    # text_units.parquet với overlap đúng
    chunks: List[Dict[str, Any]] = []
    chunk_size, overlap = 1000, 200
    step = chunk_size - overlap
    for idx, start in enumerate(range(0, len(joined_content), step), start=1):
        chunk_text = joined_content[start:start+chunk_size]
        chunks.append({
            'id': f'tu_{idx}',
            'text': chunk_text,
            'title': f'Chunk {idx}',
            'document_id': 'doc_1',
            'metadata': {'chunk_index': idx, 'length': len(chunk_text)},
        })
    text_units_df = pd.DataFrame(chunks)
    write_parquet(text_units_df, out_dir / 'text_units.parquet')
    
    # entities.parquet
    entities_rows: List[Dict[str, Any]] = []
    seen_titles: set[str] = set()
    
    def add_entity(e: Dict[str, Any]) -> None:
        title = str(e.get('name', '').strip())
        if not title:
            return
        key = normalize(title)
        if key in seen_titles:
            return
        seen_titles.add(key)
        eid = stable_id('e', title)
        meta = {
            'category': e.get('category', 'unknown'),
            'source': 'Scratch Wiki' if e.get('type') == 'block' else 'Tin học 8',
            'importance': e.get('importance', 'medium'),
            'extracted_by': 'llm',
        }
        for k in ('shape','args','aliases','aliases_vi','name_vi'):
            if e.get(k) is not None:
                meta[k] = e.get(k)
        entities_rows.append({
            'id': eid,
            'title': title,
            'description': e.get('description', ''),
            'type': e.get('type', 'concept'),
            'text_unit_ids': [c['id'] for c in chunks[:3]],
            'metadata': meta,
        })
    
    for e in block_entities:
        add_entity(e)
    for e in edu_entities:
        add_entity(e)
    
    if not entities_rows:
        for name, desc in {
            'Scratch': 'Ngôn ngữ lập trình trực quan',
            'sprite': 'Nhân vật hoạt động trong Scratch',
            'khối lệnh': 'Các thành phần cơ bản để tạo chương trình',
            'sân khấu': 'Không gian làm việc chính',
        }.items():
            entities_rows.append({
                'id': stable_id('e', name),
                'title': name,
                'description': desc,
                'type': 'concept',
                'text_unit_ids': [c['id'] for c in chunks[:3]],
                'metadata': {
                    'category': 'fallback',
                    'source': 'manual',
                    'importance': 'high',
                    'extracted_by': 'fallback',
                },
            })
    
    entities_df = pd.DataFrame(entities_rows)
    write_parquet(entities_df, out_dir / 'entities.parquet')
    
    # relationships.parquet
    rel_rows: List[Dict[str, Any]] = []
    rid = 1
    skipped_count = 0
    for r in rels:
        sid = find_entity_id_by_name(entities_df, r['source'])
        tid = find_entity_id_by_name(entities_df, r['target'])
        if not sid or not tid:
            skipped_count += 1
            if skipped_count <= 5:  # Only log first 5
                log.warning(f"Skip relationship (no match): {r['source']} -> {r['target']}")
            continue
        rel_rows.append({
            'id': f'r_{rid}',
            'source': sid,
            'target': tid,
            'description': r.get('relation', ''),
            'weight': 0.8,
            'text_unit_ids': [c['id'] for c in chunks[:3]],
            'metadata': {
                'relation_type': r.get('type', 'unknown'),
                'strength': r.get('strength', 'medium'),
                'extracted_by': 'llm',
            },
        })
        rid += 1
    
    if skipped_count > 0:
        log.info(f"⚠️  Skipped {skipped_count}/{len(rels)} relationships due to entity mismatch")
    
    if not rel_rows:
        def pick(title_sub: str) -> Optional[str]:
            m = entities_df[entities_df['title'].str.contains(re.escape(title_sub), case=False, na=False, regex=True)]
            return None if m.empty else m.iloc[0]['id']
        pairs = [
            ('Scratch', 'lập trình'),
            ('sprite', 'sân khấu'),
        ]
        for s, t in pairs:
            sid, tid = pick(s), pick(t)
            if sid and tid:
                rel_rows.append({
                    'id': f'r_{rid}',
                    'source': sid,
                    'target': tid,
                    'description': f"{s} liên quan {t}",
                    'weight': 0.8,
                    'text_unit_ids': [c['id'] for c in chunks[:3]],
                    'metadata': {
                        'relation_type': 'conceptual',
                        'strength': 'strong',
                        'extracted_by': 'fallback',
                    },
                })
                rid += 1
    
    relationships_df = pd.DataFrame(rel_rows)
    write_parquet(relationships_df, out_dir / 'relationships.parquet')
    
    # communities + reports
    ids = list(entities_df['id'])
    buckets = [ids[0:10], ids[10:20], ids[20:30], ids[30:]]
    comm_df = pd.DataFrame({
        'id': ['c_1', 'c_2', 'c_3', 'c_4'],
        'level': [1, 1, 2, 2],
        'entities': buckets,
        'metadata': [
            {'name': 'Basic Concepts', 'type': 'basic', 'color': 'blue'},
            {'name': 'Programming Concepts', 'type': 'programming', 'color': 'green'},
            {'name': 'Advanced Features', 'type': 'advanced', 'color': 'orange'},
            {'name': 'Educational Concepts', 'type': 'educational', 'color': 'purple'},
        ],
    })
    write_parquet(comm_df, out_dir / 'communities.parquet')
    
    comm_rep_df = pd.DataFrame({
        'id': ['cr_1', 'cr_2', 'cr_3', 'cr_4'],
        'community_id': ['c_1', 'c_2', 'c_3', 'c_4'],
        'title': ['Báo cáo khái niệm cơ bản', 'Báo cáo khái niệm lập trình', 'Báo cáo tính năng nâng cao', 'Báo cáo khái niệm giáo dục'],
        'summary': [
            'Tóm tắt các khái niệm cơ bản nhất trong Scratch như sprite, sân khấu, khối lệnh.',
            'Tóm tắt các khái niệm lập trình cốt lõi như vòng lặp, điều kiện, biến.',
            'Tóm tắt các tính năng nâng cao như clone, pen, sensing, video, music.',
            'Tóm tắt các khái niệm giáo dục như tư duy logic, giải quyết vấn đề, sáng tạo.',
        ],
        'metadata': [
            {'report_type': 'summary', 'generated_by': 'script'},
            {'report_type': 'summary', 'generated_by': 'script'},
            {'report_type': 'summary', 'generated_by': 'script'},
            {'report_type': 'summary', 'generated_by': 'script'},
        ],
    })
    write_parquet(comm_rep_df, out_dir / 'community_reports.parquet')
    
    log.info("== DONE ==")
    log.info(f"Documents: {len(documents_df)} | TextUnits: {len(text_units_df)} | Entities: {len(entities_df)} | Relationships: {len(relationships_df)} | Communities: {len(comm_df)} | Reports: {len(comm_rep_df)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract entities + relationships and write parquet")
    parser.add_argument("paths", nargs="*", help="Đường dẫn file .txt. Mặc định dùng /mnt/data/Blocks.txt và /mnt/data/scratch_clean.txt nếu có")
    args = parser.parse_args()
    build_kg(args.paths or None)
