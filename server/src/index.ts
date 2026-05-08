import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { api } from './routes/api.js';

const app = express();
app.use(cors());
app.use(express.json({ limit: '256kb' }));
app.use('/api', api);

app.get('/healthz', (_req, res) => res.json({ ok: true }));

app.use((err: unknown, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  const message = err instanceof Error ? err.message : 'Unknown error';
  res.status(500).json({ error: message });
});

const port = Number(process.env.PORT ?? 8787);
app.listen(port, () => {
  console.log(`[rvm-server] listening on :${port}`);
});
