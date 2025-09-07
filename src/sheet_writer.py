import os
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

SCOPE = ["https://www.googleapis.com/auth/spreadsheets"]

def _client_from_env():
    sa_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    # Use eval to parse JSON string stored in secret; alternatively json.loads
    import json
    creds = Credentials.from_service_account_info(json.loads(sa_json), scopes=SCOPE)
    return gspread.authorize(creds)

def _open_sheet(sheet_id: str):
    gc = _client_from_env()
    return gc.open_by_key(sheet_id)

def _ensure_worksheet(ss, title: str):
    try:
        return ss.worksheet(title)
    except gspread.WorksheetNotFound:
        return ss.add_worksheet(title=title, rows=200, cols=50)

def _get_header(ws):
    return ws.row_values(1) or []

def _ensure_header(ws, today_col: str):
    header = _get_header(ws)
    if not header:
        ws.update("A1", [["Hotel", today_col]])
        return
    # Ensure first col is Hotel
    if not header or header[0] != "Hotel":
        header = ["Hotel"] + header
        ws.resize(rows=ws.row_count, cols=max(len(header), ws.col_count))
        ws.update("A1", [header])
    # Ensure today's column exists
    if today_col not in header:
        ws.update_cell(1, len(header) + 1, today_col)

def _ensure_hotel_rows(ws, hotels: list[str]):
    colA = ws.col_values(1)
    existing = set(colA[1:]) if len(colA) > 1 else set()
    needed = [h for h in hotels if h not in existing]
    if needed:
        start_row = len(colA) + 1 if colA else 2
        ws.update(f"A{start_row}", [[h] for h in needed])

def _col_index(ws, col_name: str) -> int:
    header = _get_header(ws)
    return header.index(col_name) + 1

def write_results(sheet_id: str, website: str, hotels: list[str], ratings: dict[str, float], when: datetime):
    ss = _open_sheet(sheet_id)
    ws = _ensure_worksheet(ss, website.upper())
    today_col = when.strftime("%Y-%m-%d")

    _ensure_header(ws, today_col)
    _ensure_hotel_rows(ws, hotels)

    hotel_cells = ws.col_values(1)
    index_map = {name: idx + 1 for idx, name in enumerate(hotel_cells)}  # 1-based
    col = _col_index(ws, today_col)

    updates = []
    for h in hotels:
        r = ratings.get(h)
        if r is None:
            continue
        cell = gspread.utils.rowcol_to_a1(index_map[h], col)
        updates.append({"range": cell, "values": [[float(r)]]})

    if updates:
        ws.batch_update(updates, value_input_option="USER_ENTERED")

