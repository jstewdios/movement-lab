try {
 const require = (ok, msg) => { if (!ok) throw new Error(msg); };
 require(DATA.length === 35, '35 saved trials');
 require(sel.options.length === 35, '35 interactive choices');
 let snapshots = 0;
 for (let i = 0; i < DATA.length; ++i) {
   sel.value = i; sel.dispatchEvent(new Event('change'));
   require(cur === DATA[i], 'trial selection');
   for (const frame of [0, Math.floor(cur.frames.length / 2), cur.frames.length - 1]) {
     scrub.value = frame; scrub.dispatchEvent(new Event('input'));
     require(k === frame, 'timeline control');
     require(info.textContent.includes('trunk tilt'), 'rendered measurements');
     snapshots++;
   }
 }
 document.getElementById('play').click();
 require(!playing, 'pause control');
 const chosen = DATA.findIndex(d => d.label.startsWith('cem-s1, +0.32'));
 sel.value = chosen; sel.dispatchEvent(new Event('change'));
 k = cur.frames.findIndex(f => f.t >= cur.push_time + .30);
 draw();
 const result = {passed:true, engine:navigator.userAgent, trials:DATA.length, rendered_snapshots:snapshots,
   checks:['trial selection','timeline control','all three views at 105 snapshots','pause control','measured sole silhouettes'],
   screenshot_case:cur.label, screenshot_time_s:cur.frames[k].t};
 const log = document.createElement('pre'); log.id = 'browser-check'; log.textContent=JSON.stringify(result); document.body.appendChild(log);
 document.body.dataset.validation='passed';
} catch (error) {
 const log = document.createElement('pre'); log.id='browser-check'; log.textContent=JSON.stringify({passed:false,error:String(error)}); document.body.appendChild(log);
 document.body.dataset.validation='failed';
}
