import json
from pathlib import Path
ui_path = Path(r'D:\Python\open_edu_video_generator_LTX\workflows\ltxv_text_to_video_0.9.5_ui.json')
api_path = Path(r'D:\Python\open_edu_video_generator_LTX\workflows\ltx_video_12gb_api.json')
data = json.loads(ui_path.read_text(encoding='utf-8'))
links = {link[0]: (str(link[1]), link[2]) for link in data['links']}
prompt = {}
for node in data['nodes']:
    node_id = str(node['id'])
    ctype = node['type']
    if ctype == 'Note':
        continue
    item = {'class_type': ctype, 'inputs': {}}
    widget_values = list(node.get('widgets_values') or [])
    for inp in node.get('inputs') or []:
        link_id = inp.get('link')
        if link_id is not None:
            src_id, src_slot = links[link_id]
            item['inputs'][inp['name']] = [src_id, src_slot]
    if ctype == 'CheckpointLoaderSimple':
        item['inputs']['ckpt_name'] = 'ltxv-2b-0.9.8-distilled-fp8.safetensors'
    elif ctype == 'CLIPLoader':
        item['inputs']['clip_name'] = 't5xxl_fp16.safetensors'
        item['inputs']['type'] = 'ltxv'
        item['inputs']['device'] = 'default'
    elif ctype == 'CLIPTextEncode':
        title = (node.get('title') or '').lower()
        if 'negative' in title:
            item['inputs']['text'] = 'text, letters, typography, labels, captions, watermark, logo, signature, diagram, infographic, chart, collage, grid, panels, border, blurry, distorted anatomy, duplicate parts, oversaturated, bad motion, flicker, low quality, worst quality, deformed, distorted, disfigured, motion artifacts'
        else:
            item['inputs']['text'] = 'realistic educational nature video, stable camera, gentle natural motion, coherent subject, a bee slowly visiting a bright flower, pollen visible on the bee body, realistic petals and anthers, soft daylight, shallow depth of field, no text, no subtitles, no watermark'
    elif ctype == 'LTXVConditioning':
        item['inputs']['frame_rate'] = 16
    elif ctype == 'LTXVScheduler':
        vals = widget_values + [None] * 5
        item['inputs']['steps'] = 8
        item['inputs']['max_shift'] = vals[1] if vals[1] is not None else 2.05
        item['inputs']['base_shift'] = vals[2] if vals[2] is not None else 0.95
        item['inputs']['stretch'] = vals[3] if vals[3] is not None else True
        item['inputs']['terminal'] = vals[4] if vals[4] is not None else 0.1
    elif ctype == 'SamplerCustom':
        vals = widget_values + [None] * 4
        item['inputs']['add_noise'] = vals[0] if vals[0] is not None else True
        item['inputs']['noise_seed'] = 552872474466407
        item['inputs']['control_after_generate'] = 'randomize'
        item['inputs']['cfg'] = 1.0
    elif ctype == 'KSamplerSelect':
        item['inputs']['sampler_name'] = 'res_multistep'
    elif ctype == 'EmptyLTXVLatentVideo':
        item['inputs']['width'] = 768
        item['inputs']['height'] = 512
        item['inputs']['length'] = 81
        item['inputs']['batch_size'] = 1
    elif ctype == 'SaveAnimatedWEBP':
        item['inputs']['filename_prefix'] = 'video/ltx_12gb'
        item['inputs']['fps'] = 16
        item['inputs']['lossless'] = False
        item['inputs']['quality'] = 95
        item['inputs']['method'] = 'default'
    elif ctype == 'SaveWEBM':
        item['inputs']['filename_prefix'] = 'video/ltx_12gb'
        item['inputs']['codec'] = 'vp9'
        item['inputs']['fps'] = 16
        item['inputs']['crf'] = 18
    prompt[node_id] = item
api_path.write_text(json.dumps(prompt, indent=2), encoding='utf-8')
print(api_path)
print(len(prompt))
