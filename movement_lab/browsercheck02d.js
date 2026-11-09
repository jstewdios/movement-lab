// Append after the replay's inline script in a QA copy, never alter its data.
try {
  const require = (ok,msg) => { if (!ok) throw new Error(msg); };
  require(trials.length >= 36, 'all declared recorded comparisons present');
  require($('left').options.length === trials.length, 'left choices');
  require($('right').options.length === trials.length, 'right choices');
  let snapshots = 0;
  for (let i = 0; i < DATA.presets.length; i++) {
    $('preset').value = i; $('preset').dispatchEvent(new Event('change'));
    require(Number($('left').value) === DATA.presets[i].left, 'paired selection');
    for (const angle of ['side','front','top']) {
      $('view').value = angle; $('view').dispatchEvent(new Event('change'));
      for (const t of [-.4, .3, Number($('scrub').max)]) {
        $('scrub').value = t; $('scrub').dispatchEvent(new Event('input'));
        require(Math.abs(elapsed - Number($('scrub').value)) < .001, 'timeline');
        require($('leftMetrics').textContent.includes('Body tilt'), 'left measurements');
        require($('rightMetrics').textContent.includes('Foot contact'), 'right measurements');
        require($('leftStatus').textContent.includes('recovery'), 'whole-episode outcome');
        snapshots++;
      }
    }
  }
  for (let i = 0; i < trials.length; i++) {
    $('left').value = i; $('left').dispatchEvent(new Event('change'));
    require(selected('left') === trials[i], 'manual choice');
    require(trials[i].frames.every((f,j,a) => j === 0 || f.t > a[j-1].t), 'recording timestamps');
  }
  $('play').click(); require(playing, 'play');
  $('play').click(); require(!playing, 'pause');
  $('restart').click(); require(elapsed === Number($('scrub').min), 'restart');
  $('view').value = 'side'; $('view').dispatchEvent(new Event('change'));
  $('preset').value = 0; $('preset').dispatchEvent(new Event('change'));
  $('scrub').value = .3; $('scrub').dispatchEvent(new Event('input'));
  const result = {passed:true, engine:navigator.userAgent, trials:trials.length,
    presets:DATA.presets.length, snapshots, controls:['paired and manual selection','timeline','three cameras','play/pause/restart','measurements','outcome labels']};
  const log=document.createElement('pre');log.id='browser-check';log.textContent=JSON.stringify(result);
  log.style.display='none';document.body.append(log);document.body.dataset.validation='passed';
} catch(error) {
  const log=document.createElement('pre');log.id='browser-check';log.textContent=JSON.stringify({passed:false,error:String(error)});
  document.body.append(log);document.body.dataset.validation='failed';
}
