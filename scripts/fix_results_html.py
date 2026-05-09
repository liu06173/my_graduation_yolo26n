"""Fix renderResults to show tracking unique counts"""
import re

f = 'E:/claude/end001/my_graduation_yolo26n/webapp/templates/index.html'
t = open(f, encoding='utf-8').read()

# 1. Add isTracking before vidHtml
t = t.replace(
    "        const vidHtml = r.output_video_name",
    "        const isTracking = r.mode === 'tracking';\n        const vidHtml = r.output_video_name"
)

# 2. Add uniqueHtml block before the detection count row
old_block = '${vidHtml}\n            <div class="metric-row"><span>'
new_block = '''${vidHtml}
            ${uniqueHtml}
            <div class="metric-row"><span>'''

if '${uniqueHtml}' not in t:
    t = t.replace(old_block, new_block)

# 3. Add uniqueHtml definition between vidHtml closing and before the mode label
# Find the spot right after vidHtml definition and before "const modeLabel"
t = t.replace(
    "            : '';\n        const modeLabel",
    '''            : '';
        // 跟踪模式：显示独立目标数
        const uniqueHtml = isTracking
            ? `<div style="margin:8px 0;padding:12px;background:rgba(52,211,153,0.08);border:1px solid rgba(52,211,153,0.2);border-radius:8px;">
                 <div style="font-size:0.9em;font-weight:700;color:var(--green);margin-bottom:6px;">独立目标数 (跟踪ID去重)</div>
                 <div style="display:flex;gap:16px;flex-wrap:wrap;">
                   ${Object.entries(r.unique_by_class||{}).map(([k,v]) => `
                     <div style="text-align:center;">
                       <div style="font-size:1.4em;font-weight:800;color:var(--green);">${v}</div>
                       <div style="font-size:0.78em;color:var(--text2);">${k}</div>
                     </div>
                   `).join('')}
                 </div>
                 <div style="font-size:0.78em;color:var(--text2);margin-top:4px;">共 <b style="color:var(--green);">${r.unique_objects}</b> 个独立目标</div>
               </div>`
            : '';
        const modeLabel'''
)

# 4. Change total_objects to total_detections||total_objects for tracking compat
t = t.replace(
    '${r.total_objects}',
    '${r.total_detections||r.total_objects}'
)

# 5. Add mode label to card header
t = t.replace(
    '<h3>${r.weight} <span style="color:var(--text2);font-weight:400;">(${r.model})</span></h3>',
    '<h3>${r.weight} <span style="color:var(--text2);font-weight:400;">(${r.model})</span> <span style="font-size:0.75em;color:var(--blue);">${isTracking ? "跟踪统计" : "逐帧检测"}</span></h3>'
)

open(f, 'w', encoding='utf-8').write(t)
print('DONE - renderResults updated for tracking mode')
