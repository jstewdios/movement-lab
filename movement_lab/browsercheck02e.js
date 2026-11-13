// QA only: appended to a copy of the replay; never alters its data.
(function(){const out={passed:false};try{
  const need=(ok,msg)=>{if(!ok)throw new Error(msg)};
  need(trials.length===26,'all declared recordings present: '+trials.length);
  need(DATA.presets.length===9,'all declared presets present: '+DATA.presets.length);
  for(let k=0;k<3;k++)need($('pick'+k).options.length===trials.length,'manual choices '+k);
  let snapshots=0;
  for(let i=0;i<DATA.presets.length;i++){
    $('preset').value=i;$('preset').dispatchEvent(new Event('change'));
    for(let k=0;k<3;k++)need(Number($('pick'+k).value)===DATA.presets[i].panels[k],'preset selection '+i);
    for(const v of ['side','front','top']){
      $('view').value=v;$('view').dispatchEvent(new Event('change'));
      for(const t of [Number($('scrub').min),-.5,.3,Number($('scrub').max)]){
        $('scrub').value=t;$('scrub').dispatchEvent(new Event('input'));
        need(Math.abs(elapsed-Number($('scrub').value))<.001,'timeline');
        for(let k=0;k<3;k++){
          need($('metrics'+k).textContent.includes('Perceived pitch'),'measurements '+k);
          need(/recovery|Failed/.test($('status'+k).textContent),'outcome label '+k);
          const d=trials[Number($('pick'+k).value)];
          need($('status'+k).textContent.startsWith(d.recovered?'Sustained recovery':'Failed'),'label matches audit');
        }
        snapshots++;
      }
    }
  }
  $('pick1').value=3;$('pick1').dispatchEvent(new Event('change'));
  $('play').click();$('play').click();$('restart').click();
  out.passed=true;out.trials=trials.length;out.presets=DATA.presets.length;out.snapshots=snapshots;
  out.controls=['preset and manual selection','timeline','three cameras','play/pause/restart','measurements','outcome labels'];
}catch(e){out.error=String(e)}
const pre=document.createElement('pre');pre.id='qa';pre.textContent=JSON.stringify(out);document.body.append(pre);})();
