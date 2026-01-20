/**
 * Geospatial Penetration Routes
 * Serves processed Aadhaar penetration and EEI data from CSV
 */

import express from 'express';
import fs from 'fs';
import path from 'path';
import csvParser from 'csv-parser';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const router = express.Router();

// Paths to CSV files
const PENETRATION_FILE = path.join(__dirname, '../geospatial-data/features/penetration_by_state.csv');
const EEI_FILE = path.join(__dirname, '../geospatial-data/features/enrollment_efficiency_by_state.csv');

/**
 * Load and parse CSV file
 */
function loadCSV(filePath) {
    return new Promise((resolve, reject) => {
        if (!fs.existsSync(filePath)) {
            reject(new Error(`File not found: ${filePath}`));
            return;
        }

        const records = [];

        fs.createReadStream(filePath)
            .pipe(csvParser())
            .on('data', (row) => records.push(row))
            .on('end', () => resolve(records))
            .on('error', (error) => reject(error));
    });
}

/**
 * GET /api/geo-penetration/state
 * Returns state-level penetration data
 */
router.get('/state', async (req, res) => {
    console.log('🗺️  Geo-Penetration API: /state endpoint hit');
    try {
        console.log('📂 Loading CSV from:', PENETRATION_FILE);
        const data = await loadCSV(PENETRATION_FILE);

        const result = data.map(row => ({
            state: row.STATE,
            penetration_pct: parseFloat(row.PENETRATION_PCT) || null,
            total_enrollment: parseInt(row.TOTAL_ENROLLMENT) || 0,
            population: parseInt(row.POPULATION) || null
        }));

        console.log(`📊 Geo-Penetration API: Returning ${result.length} states`);
        res.json(result);

    } catch (error) {
        console.error('❌ Geo-Penetration API Error:', error.message);
        res.status(500).json({
            error: 'Failed to load penetration data',
            message: error.message
        });
    }
});

/**
 * GET /api/geo-penetration/eei
 * Returns Enrollment Efficiency Index data
 */
router.get('/eei', async (req, res) => {
    try {
        const data = await loadCSV(EEI_FILE);

        const result = data.map(row => ({
            state: row.STATE,
            actual_enrollment: parseInt(row.ACTUAL_ENROLLMENT) || 0,
            expected_enrollment: parseInt(row.EXPECTED_ENROLLMENT) || null,
            eei: parseFloat(row.EEI) || null
        }));

        // Compute statistics
        const withEEI = result.filter(r => r.eei !== null);
        const withoutEEI = result.filter(r => r.eei === null);

        console.log(`📊 EEI API: Returning ${result.length} states (${withEEI.length} with EEI, ${withoutEEI.length} missing)`);

        res.json({
            success: true,
            count: result.length,
            statesWithEEI: withEEI.length,
            statesMissingEEI: withoutEEI.length,
            data: result
        });

    } catch (error) {
        console.error('❌ EEI API Error:', error.message);
        res.status(500).json({
            success: false,
            error: 'Failed to load EEI data',
            message: error.message
        });
    }
});

export default router;
