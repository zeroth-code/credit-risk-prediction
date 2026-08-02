// Capture a README screenshot from a running Streamlit demonstration.
// Usage: node scripts/capture_screenshot.mjs <url> <output-path>
import { chromium } from 'playwright';

const [url, output] = process.argv.slice(2);
if (!url || !output) {
  console.error('usage: node scripts/capture_screenshot.mjs <url> <output-path>');
  process.exit(1);
}

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1600, height: 1100 }, deviceScaleFactor: 2 });
page.on('pageerror', (error) => console.error(`pageerror: ${error.message}`));

await page.goto(url, { waitUntil: 'networkidle', timeout: 120000 });
await page.waitForSelector('[data-testid="stAppViewContainer"]', { timeout: 120000 });
await page.waitForTimeout(6000);

// Score the pre-populated synthetic application so the capture shows a real result.
const run = page.getByRole('button', { name: /run assessment/i });
if (await run.count()) {
  await run.first().click();
  await page.waitForFunction(
    () => !document.body.innerText.includes('Run the synthetic application to view'),
    { timeout: 120000 },
  );
  await page.waitForTimeout(4000);
}

const text = await page.innerText('body');
for (const marker of ['Traceback', 'StartupError', 'release bundle is unavailable']) {
  if (text.includes(marker)) {
    console.error(`page reported a failure marker: ${marker}`);
    await browser.close();
    process.exit(2);
  }
}

await page.screenshot({ path: output });
console.log(`captured ${output}`);
console.log(text.split('\n').filter(Boolean).slice(0, 12).join('\n'));
await browser.close();
