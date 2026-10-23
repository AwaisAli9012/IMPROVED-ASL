function d(n){const t=n-1;return t*t*t+1}function r(n){const t=typeof n=="string"&&n.match(/^\s*(-?[\d.]+)([^\s]*)\s*$/);return t?[parseFloat(t[1]),t[2]||"px"]:[n,"px"]}function g(n,{delay:t=0,duration:a=400,easing:e=d,x:i=0,y:u=0,opacity:f=0}={}){const s=getComputedStyle(n),o=+s.opacity,p=s.transform==="none"?"":s.transform,y=o*(1-f),[_,l]=r(i),[$,m]=r(u);return{delay:t,duration:a,easing:e,css:(c,x)=>`
			transform: ${p} translate(${(1-c)*_}${l}, ${(1-c)*$}${m});
			opacity: ${o-y*x}`}}export{g as f};
