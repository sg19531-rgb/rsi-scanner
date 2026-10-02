# app.py - NSE Microcap Monthly Wilder RSI(14) 68-72 Scanner
# pip install flask yfinance pandas numpy requests
# run: python app.py -> http://0.0.0.0:5000  dashboard at / (static/index.html)
from flask import Flask, jsonify, request, send_from_directory
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import time
import threading
import random
import os
import json
import traceback
import requests

app = Flask(__name__, static_folder='static', static_url_path='')

FALLBACK_MICROCAPS = ["ASKAUTOLTD","AURIONPRO","ADFFOODS","AHLADA","AIRAN","AJMERA","ALBERTDAV","ALEMBICLTD","AMIABLE","AMIORG","ANUP","APOLLOPIPE","ARMANFIN","ASHAPURMIN","ASMTEC","ASTRON","ATUL","AURUM","AVADHSUGAR","AXISCADES","BAJAJHCARE","BALAJITELE","BANARISUG","BANSWRAS","BARTRONICS","BEARDSELL","BFINVEST","BHAGYANGR","BHARATRAS","BIGBLOC","BINNY","BIOFILCHEM","BIRLACABLE","BLACKROSE","BLS","BODALCHEM","BORORENEW","BUTTERFLY","CAMLINFINE","CAPACITE","CAPLIPOINT","CARERATING","CARYSIL","CENTENKA","CENTRUM","CHEMBOND","CHEMFAB","CHOICEIN","CIGNITITEC","CINEVISTA","COFFEEDAY","CONCORDBIO","CONTROLPR","COSMOFIRST","CREST","CYBERMEDIA","CYBERTECH","DBCORP","DBREALTY","DCMSHRIRAM","DCW","DEN","DHANBANK","DHUNINV","DIAMINESQ","DICIND","DOLATALGO","DOLLAR","DREDGECORP","DSSL","DYNAMATECH","E2E","EASEMYTRIP","ECEIND","ELGIEQUIP","ENIL","ERIS","ESABINDIA","EXPLEOSOL","FAIRCHEMOR","FDC","FIEMIND","FILATEX","FINCABLES","FINPIPE","FLEXITUFF","FOSECOIND","GABRIEL","GALAXYSURF","GANDHAR","GANGAFORGE","GARFIBRES","GENESYS","GENUSPAPER","GEOJITFSL","GHCLTEXTIL","GICHSGFIN","GILLANDERS","GLOBALVECT","GLOBEINT","GODFRYPHLP","GOKEX","GOLDIAM","GOODLUCK","GOODYEAR","GPIL","GREENLAM","GREENPANEL","GREENPLY","GRPLTD","GTPL","GUFICBIO","GUJALKALI","GUJAPOLLO","GUJTERC","HARSHA","HATHWAY","HBLPOWER","HCC","HCG"]

STATE = {"scanning": False, "progress": 0, "total": 0, "done": 0, "last_scan_utc": None, "last_duration_s": None, "universe_count": 0, "matches_count": 0, "universe_src": "fallback_json", "error": None}
RESULTS = {"as_of_utc": None, "universe_count": 0, "universe_src": "fallback_json", "matches": [], "verification": {}, "duration_s": None}
CHART_CACHE = {}

def utc_today():
    return datetime.datetime.now(datetime.timezone.utc).date()

def flatten_yf(df):
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        try:
            l0 = [str(x).strip().lower() for x in df.columns.get_level_values(0)]
            l1 = [str(x).strip().lower() for x in df.columns.get_level_values(1)]
            ohlc = {'open','high','low','close','adj close','volume'}
            s0 = set(l0)
            s1 = set(l1)
            if s0.intersection(ohlc):
                df.columns = [str(c).strip() for c in df.columns.get_level_values(0)]
            elif s1.intersection(ohlc):
                df.columns = [str(c).strip() for c in df.columns.get_level_values(1)]
            else:
                df.columns = [str(c[0]).strip() for c in df.columns]
        except Exception:
            try:
                df.columns = [str(c[0]).strip() for c in df.columns]
            except Exception:
                pass
    else:
        df.columns = [str(c).strip() for c in df.columns]
    mapping = {}
    for c in list(df.columns):
        lc = str(c).lower().strip()
        if lc == 'open':
            mapping[c] = 'Open'
        elif lc == 'high':
            mapping[c] = 'High'
        elif lc == 'low':
            mapping[c] = 'Low'
        elif lc == 'close':
            mapping[c] = 'Close'
        elif lc in ('adj close','adjclose','adj_close'):
            mapping[c] = 'Adj Close'
        elif lc == 'volume':
            mapping[c] = 'Volume'
    if mapping:
        df = df.rename(columns=mapping)
    try:
        df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    except Exception:
        pass
    return df

def wilder_rsi(close, period=14):
    close = pd.Series(close)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    flat = (avg_gain == 0) & (avg_loss == 0)
    rsi[flat] = 50.0
    return rsi

def get_universe():
    try:
        total_url = 'https://www.niftyindices.com/IndexConstituent/ind_nifttotalmarket_list.csv'
        n500_url = 'https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv'
        headers = {'User-Agent': 'Mozilla/5.0'}
        t = requests.get(total_url, headers=headers, timeout=8)
        n = requests.get(n500_url, headers=headers, timeout=8)
        if t.status_code == 200 and n.status_code == 200:
            import csv
            import io
            def parse_symbols(text):
                syms = set()
                try:
                    f = io.StringIO(text)
                    r = csv.DictReader(f)
                    if r.fieldnames:
                        sym_key = None
                        for k in r.fieldnames:
                            if k and k.strip().lower() == 'symbol':
                                sym_key = k
                                break
                        if sym_key:
                            for row in r:
                                s = str(row.get(sym_key,'')).strip().upper()
                                if s:
                                    syms.add(s)
                except Exception:
                    pass
                return syms
            total = parse_symbols(t.text)
            n500 = parse_symbols(n.text)
            micro = sorted(list(total - n500))
            if len(micro) >= 50:
                return micro, 'live:total_minus_500'
    except Exception:
        pass
    try:
        if os.path.exists('fallback_universe.json'):
            with open('fallback_universe.json','r') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) >= 20:
                    syms = [str(x).strip().upper().replace('.NS','') for x in data if str(x).strip()]
                    return sorted(list(set(syms))), 'fallback_json'
                if isinstance(data, dict) and 'symbols' in data:
                    syms = [str(x).strip().upper().replace('.NS','') for x in data['symbols']]
                    return sorted(list(set(syms))), 'fallback_json'
    except Exception:
        pass
    return sorted(FALLBACK_MICROCAPS), 'fallback_json'

def yf_download(ticker, interval):
    today = utc_today()
    tomorrow = today + datetime.timedelta(days=1)
    start = (today - datetime.timedelta(days=365*9)).isoformat()
    end = tomorrow.isoformat()
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False, threads=False, actions=False)
    df = flatten_yf(df)
    return df

def expected_session():
    today = utc_today()
    wd = today.weekday()
    if wd == 5:
        return today - datetime.timedelta(days=1)
    if wd == 6:
        return today - datetime.timedelta(days=2)
    return today

def verify_aska():
    try:
        df = yf_download('ASKAUTOLTD.NS', '1d')
        if df is None or df.empty:
            return {'symbol':'ASKAUTOLTD','verified':False,'reason':'empty'}
        last_ts = df.index[-1]
        try:
            if getattr(last_ts, 'tzinfo', None) is not None:
                last_date = pd.Timestamp(last_ts).tz_convert(None).date()
            else:
                last_date = pd.Timestamp(last_ts).date()
        except Exception:
            last_date = pd.Timestamp(last_ts).date()
        exp = expected_session()
        today = utc_today()
        return {'symbol':'ASKAUTOLTD','last_daily_date':last_date.isoformat(),'expected_session':exp.isoformat(),'today_utc':today.isoformat(),'verified':(last_date == exp)}
    except Exception as e:
        return {'symbol':'ASKAUTOLTD','verified':False,'error':str(e)}

def scan_once():
    t0 = time.time()
    universe, src = get_universe()
    STATE.update({'scanning':True,'progress':0,'total':len(universe),'done':0,'universe_count':len(universe),'universe_src':src,'error':None})
    matches = []
    for i, base in enumerate(universe):
        ticker = base if str(base).endswith('.NS') else str(base)+'.NS'
        try:
            df = yf_download(ticker, '1mo')
            if df is not None and not df.empty and len(df) >= 16:
                col = None
                if 'Close' in df.columns:
                    col = 'Close'
                elif 'Adj Close' in df.columns:
                    col = 'Adj Close'
                if col is not None:
                    closes = df[col].dropna()
                    if len(closes) >= 15:
                        rsi = wilder_rsi(closes, 14)
                        last_rsi = float(rsi.iloc[-1])
                        if 68.0 <= last_rsi <= 72.0:
                            price = float(closes.iloc[-1])
                            prev = float(closes.iloc[-2]) if len(closes) >= 2 else price
                            chg = (price - prev)/prev*100 if prev else 0.0
                            vol = None
                            if 'Volume' in df.columns:
                                try:
                                    vol = float(df['Volume'].iloc[-1])
                                except Exception:
                                    vol = None
                            try:
                                last_month_str = pd.Timestamp(df.index[-1]).date().isoformat()
                            except Exception:
                                last_month_str = str(df.index[-1])[:10]
                            sym_base = str(base).replace('.NS','').upper()
                            matches.append({'symbol':sym_base,'ticker':ticker,'price':round(price,2),'monthly_rsi':round(last_rsi,2),'monthly_change_pct':round(float(chg),2),'volume':vol,'last_month':last_month_str})
        except Exception:
            pass
        STATE['done'] = i+1
        if universe:
            STATE['progress'] = round((i+1)/len(universe)*100,1)
    matches = sorted(matches, key=lambda x: x['monthly_rsi'], reverse=True)
    verification = verify_aska()
    as_of = datetime.datetime.now(datetime.timezone.utc).isoformat()
    dur = round(time.time()-t0,1)
    RESULTS.update({'as_of_utc':as_of,'universe_count':len(universe),'universe_src':src,'matches':matches,'verification':verification,'duration_s':dur})
    STATE.update({'scanning':False,'progress':100,'last_scan_utc':as_of,'last_duration_s':dur,'matches_count':len(matches)})
    return RESULTS

def resample_ohlc(df_daily, rule):
    agg = {}
    if 'Open' in df_daily.columns:
        agg['Open'] = 'first'
    if 'High' in df_daily.columns:
        agg['High'] = 'max'
    if 'Low' in df_daily.columns:
        agg['Low'] = 'min'
    if 'Close' in df_daily.columns:
        agg['Close'] = 'last'
    elif 'Adj Close' in df_daily.columns:
        agg['Adj Close'] = 'last'
    if 'Volume' in df_daily.columns:
        agg['Volume'] = 'sum'
    out = df_daily.resample(rule).agg(agg)
    close_col = 'Close' if 'Close' in out.columns else 'Adj Close'
    out = out.dropna(subset=[close_col])
    return out

def get_chart_data(symbol_base, tf):
    tf = str(tf).upper() if tf else 'D'
    if tf not in ('D','W','M'):
        tf = 'D'
    key = str(symbol_base).upper()+':'+tf
    now = time.time()
    if key in CHART_CACHE:
        ts, ttl, data = CHART_CACHE[key]
        if now - ts < ttl:
            return data
    ticker = symbol_base if str(symbol_base).endswith('.NS') else str(symbol_base)+'.NS'
    df = yf_download(ticker, '1d')
    if df is None or df.empty:
        raise ValueError('no daily data')
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='last')]
    if tf == 'W':
        df = resample_ohlc(df, 'W-FRI')
    elif tf == 'M':
        df = resample_ohlc(df, 'ME')
    today = utc_today()
    today_ts = pd.Timestamp(today)
    if not df.empty and tf in ('W','M'):
        try:
            idx = list(df.index)
            idx[-1] = today_ts
            df.index = idx
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='last')]
        except Exception:
            pass
    if 'Close' in df.columns:
        closes = df['Close']
    else:
        closes = df['Adj Close']
    ema21 = closes.ewm(span=21, adjust=False).mean()
    rsi = wilder_rsi(closes, 14)
    df = df.tail(300)
    closes = closes.tail(300)
    ema21 = ema21.tail(300)
    rsi = rsi.tail(300)
    candles = []
    for ts_idx, row in df.iterrows():
        try:
            d = pd.Timestamp(ts_idx).date().isoformat()
        except Exception:
            d = str(ts_idx)[:10]
        o = round(float(row['Open']),2) if 'Open' in df.columns and pd.notna(row['Open']) else None
        h = round(float(row['High']),2) if 'High' in df.columns and pd.notna(row['High']) else None
        l = round(float(row['Low']),2) if 'Low' in df.columns and pd.notna(row['Low']) else None
        c = round(float(row['Close']),2) if 'Close' in df.columns and pd.notna(row['Close']) else (round(float(row['Adj Close']),2) if 'Adj Close' in df.columns else None)
        v = float(row['Volume']) if 'Volume' in df.columns and pd.notna(row['Volume']) else 0
        candles.append({'time':d,'open':o,'high':h,'low':l,'close':c,'volume':v})
    ema_list = []
    rsi_list = []
    for ts_idx in df.index:
        try:
            d = pd.Timestamp(ts_idx).date().isoformat()
        except Exception:
            d = str(ts_idx)[:10]
        try:
            ev = float(ema21.loc[ts_idx])
            if not np.isnan(ev):
                ema_list.append({'time':d,'value':round(ev,2)})
        except Exception:
            pass
        try:
            rv = float(rsi.loc[ts_idx])
            if not np.isnan(rv):
                rsi_list.append({'time':d,'value':round(rv,2)})
        except Exception:
            pass
    last_close = float(closes.iloc[-1]) if len(closes) else None
    last_rsi = None
    try:
        lr = float(rsi.iloc[-1])
        if not np.isnan(lr):
            last_rsi = lr
    except Exception:
        pass
    data = {'symbol':str(symbol_base).replace('.NS','').upper(),'tf':tf,'candles':candles,'ema21':ema_list,'rsi':rsi_list,'last_close':last_close,'last_rsi':last_rsi,'as_of_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    ttl = random.randint(60,180)
    CHART_CACHE[key] = (now, ttl, data)
    return data

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/index.html')
def index2():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/status')
def api_status():
    return jsonify({'ok':True,'scanning':STATE['scanning'],'progress':STATE['progress'],'total':STATE['total'],'done':STATE['done'],'last_scan_utc':STATE['last_scan_utc'],'last_duration_s':STATE['last_duration_s'],'universe_count':STATE['universe_count'],'matches_count':STATE['matches_count'],'universe_src':STATE['universe_src'],'error':STATE['error']})

@app.route('/api/scan', methods=['POST'])
def api_scan():
    body = request.get_json(silent=True) or {}
    force = bool(body.get('force', False))
    if STATE['scanning']:
        return jsonify({'started':False,'reason':'scan_in_progress','status':STATE}), 202
    def run():
        try:
            scan_once()
        except Exception as e:
            STATE.update({'scanning':False,'error':str(e)})
            traceback.print_exc()
    import threading as _th
    _th.Thread(target=run, daemon=True).start()
    return jsonify({'started':True,'force':force})

@app.route('/api/results')
def api_results():
    resp = jsonify(RESULTS)
    resp.headers['Cache-Control'] = 'public, max-age=30'
    return resp

@app.route('/api/chart/<symbol>')
def api_chart(symbol):
    tf = request.args.get('tf','D').upper()
    if tf not in ('D','W','M'):
        return jsonify({'error':'tf must be D,W,M'}), 400
    base = str(symbol).upper().strip()
    try:
        data = get_chart_data(base, tf)
        resp = jsonify(data)
        resp.headers['Cache-Control'] = 'public, max-age=30'
        return resp
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error':str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
