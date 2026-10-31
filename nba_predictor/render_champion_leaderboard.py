#!/usr/bin/env python3
"""Render the completed date search as a local, filterable HTML report."""
import argparse
import json
from pathlib import Path

import pandas as pd


def render(directory: Path):
    table = pd.read_csv(directory/'final_holdout_leaderboard.csv')
    manifest = json.loads((directory/'search_manifest.json').read_text())
    payload = table.to_json(orient='records').replace('</', '<\\/')
    details = json.dumps({key: manifest[key] for key in ('Completed_Fits', 'Date_Window_Combinations')})
    document = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NBA champion date-window leaderboard</title>
<style>
body{font:16px system-ui,sans-serif;background:#f5f7fb;color:#16213a;margin:0;padding:32px}
main{max-width:1400px;margin:auto}h1{margin-bottom:8px}p{line-height:1.6;max-width:1050px}
.cards{display:flex;gap:18px;flex-wrap:wrap;margin:24px 0}.card{background:white;padding:18px 26px;border-radius:12px;border:1px solid #dfe4ed}
.card strong{display:block;font-size:28px}.controls{display:flex;gap:20px;align-items:center;flex-wrap:wrap;margin:24px 0}
select,input{font:inherit;padding:8px;border:1px solid #aab6cb;border-radius:6px}input{width:70px}
.table{overflow:auto;background:white;border:1px solid #dfe4ed;border-radius:12px;max-height:70vh}
table{border-collapse:collapse;width:100%;white-space:nowrap}th,td{text-align:right;padding:12px 16px;border-bottom:1px solid #e8ecf2}
th{position:sticky;top:0;background:#16213a;color:white}td:nth-child(2),th:nth-child(2){text-align:left}
tbody tr:hover{background:#eef4ff}.note{font-size:14px;color:#475670}a{color:#185ad5}
</style><main>
<h1>Champion date-window leaderboard</h1>
<p>Train once on each candidate window, validate on the following seasons, then test on later years.
This table uses the same final validation end for every row. Ranks use validation results only;
the later holdout results never affect selection.</p>
<div class="cards"><div class="card"><strong id="fits"></strong>models fitted</div>
<div class="card"><strong id="windows"></strong>date combinations evaluated</div>
<div class="card"><strong id="cutoff"></strong>final validation end</div>
<div class="card"><strong id="holdout"></strong>holdout period</div></div>
<div class="controls"><label>Model <select id="model"><option value="">Both</option><option>logistic</option><option>xgboost</option></select></label>
<label>Minimum validation seasons <input id="minimum" type="number" min="1" value="1"></label>
<span id="count"></span></div>
<p class="note">Ranking: validation accuracy’s Wilson lower bound, then top-one/top-three accuracy and reciprocal rank.
This rewards stronger evidence over a single lucky season. Validation spans still differ across rows.
Many-window selection can overfit; the final holdout is only one season in the default search.</p>
<div class="table"><table><thead><tr><th>Rank</th><th>Model</th><th>Train</th><th>Validate</th><th>Seasons</th>
<th>Winner accuracy</th><th>Top-three accuracy</th><th>Wilson lower bound</th><th>Later test winner accuracy</th></tr></thead><tbody id="rows"></tbody></table></div>
<p><a href="all_windows_leaderboard.csv">All date windows (CSV)</a> ·
<a href="final_holdout_leaderboard.csv">This leaderboard (CSV)</a> ·
<a href="winners_by_validation_end.csv">Best setup at each validation end</a> ·
<a href="all_season_predictions.csv">Every season’s top-three picks</a></p>
<p class="note">Models use same-season regular-season team/player stats and strictly prior experience proxies.
These are retrospective predictions after the regular season, not preseason forecasts.</p>
</main><script>
const data=__DATA__, details=__DETAILS__;
document.querySelector('#fits').textContent=details.Completed_Fits.toLocaleString();
document.querySelector('#windows').textContent=details.Date_Window_Combinations.toLocaleString();
document.querySelector('#cutoff').textContent=data[0].Validation_End;
document.querySelector('#holdout').textContent=data[0].Later_Test_Start===data[0].Later_Test_End?data[0].Later_Test_End:data[0].Later_Test_Start+'–'+data[0].Later_Test_End;
const pct=x=>x===null?'Unscored':(100*x).toFixed(1)+'%';
function update(){const model=document.querySelector('#model').value, min=Number(document.querySelector('#minimum').value)||1;
const filtered=data.filter(r=>(!model||r.Model===model)&&r.Validation_Seasons>=min);
document.querySelector('#count').textContent=filtered.length.toLocaleString()+' configurations';
const body=document.querySelector('#rows');body.replaceChildren();
for(const r of filtered){const tr=document.createElement('tr');
for(const value of [r.Rank,r.Model,r.Train_Start+'–'+r.Train_End,r.Validation_Start+'–'+r.Validation_End,r.Validation_Seasons,
pct(r.Validation_Top1),pct(r.Validation_Top3),pct(r.Validation_Wilson_Lower),pct(r.Later_Test_Top1)]){
const td=document.createElement('td');td.textContent=value;tr.appendChild(td)}body.appendChild(tr)}}
document.querySelector('#model').addEventListener('change',update);document.querySelector('#minimum').addEventListener('input',update);update();
</script></html>'''
    path = directory/'leaderboard.html'
    path.write_text(document.replace('__DATA__', payload).replace('__DETAILS__', details), encoding='utf-8')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='?', default=Path('outputs/nba_champion_window_search_top2_ratings'))
    print(render(parser.parse_args().directory).resolve())
