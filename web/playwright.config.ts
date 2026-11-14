// The pages in real Chrome: pnpm run web-check. The system's Chrome, no browser is downloaded. A failed
// test leaves a trace with its console and requests, and a screenshot, in build/web/test-results/.

import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  outputDir: "../build/web/test-results",
  fullyParallel: true,
  workers: 3,
  forbidOnly: true,
  reporter: [["list"], ["html", { outputFolder: "../build/web/test-report", open: "never" }]],
  use: {
    // The system's Chrome by default; SOTAIGEN_CHROME_CHANNEL names another, or empty for Playwright's own Chromium.
    channel: (process.env.SOTAIGEN_CHROME_CHANNEL ?? "chrome") || undefined,
    headless: true,
    viewport: { width: 1280, height: 800 },
    // The browser's language picks the page's, and the cases read the Chinese words. The English page
    // has its own cases, which ask for it.
    locale: "zh-CN",
    acceptDownloads: true,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    // With a stand-in for the Python worker: what the page does, in seconds.
    { name: "ui", testMatch: /ui\.spec\.ts/, timeout: 30000 },
    // With the real worker, on real bodies. Few, because each makes a real body.
    { name: "smoke", testMatch: /smoke\.spec\.ts/, timeout: 300000 },
  ],
});
