// File: /pages/api/predict.js

import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';

export default async function handler(req, res) {
  const csvFile = path.join(process.cwd(), 'satta_data_with_predictions.csv');
  const scriptFile = path.join(process.cwd(), 'predict.py');

  try {
    if (!fs.existsSync(scriptFile)) {
      return res.status(404).json({ error: 'Python script not found' });
    }

    const python = spawn('python3', [scriptFile]);

    let output = '';
    let error = '';

    python.stdout.on('data', (data) => {
      output += data.toString();
    });

    python.stderr.on('data', (data) => {
      error += data.toString();
    });

    python.on('close', (code) => {
      if (code === 0) {
        const fileExists = fs.existsSync(csvFile);
        return res.status(200).json({ success: true, output, fileExists });
      } else {
        return res.status(500).json({ success: false, error });
      }
    });
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
}
