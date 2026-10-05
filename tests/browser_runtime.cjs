/* Shared browser selection. CI uses Playwright's pinned Chromium; an installed
 * Chrome/Edge can be explicitly selected for local development. */
function browserLaunchOptions(){
  const options={headless:true};
  if(process.env.CHROME_EXECUTABLE)options.executablePath=process.env.CHROME_EXECUTABLE;
  else{
    const channel=process.env.BROWSER_CHANNEL||'chromium';
    if(channel!=='chromium')options.channel=channel;
  }
  return options;
}
module.exports={browserLaunchOptions};
