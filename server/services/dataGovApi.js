import axios from 'axios';

const BASE_URL = 'https://api.data.gov.in/resource';

// Resource IDs
const RESOURCES = {
    ENROLMENT: 'ecd49b12-3084-4521-8f7e-ca8bf72069ba',
    DEMOGRAPHIC: '19eac040-0b94-49fa-b239-4f2fd8677d53',
    BIOMETRIC: '65454dab-1517-40a3-ac1d-47d4dfe6891c'
};

/**
 * Fetch data from data.gov.in API
 * @param {string} resourceId - The resource ID to fetch
 * @param {object} options - Query options (limit, offset, filters)
 */
async function fetchData(resourceId, options = {}) {
    const API_KEY = process.env.DATA_GOV_API_KEY;
    const { limit = 100, offset = 0, filters = {} } = options;

    const params = new URLSearchParams({
        'api-key': API_KEY,
        format: 'json',
        limit: limit.toString(),
        offset: offset.toString(),
        ...filters
    });

    const url = `${BASE_URL}/${resourceId}?${params}`;

    try {
        const response = await axios.get(url);
        return {
            success: true,
            total: response.data.total,
            count: response.data.count,
            records: response.data.records || []
        };
    } catch (error) {
        console.error(`API Error for ${resourceId}:`, error.message);
        return {
            success: false,
            error: error.message,
            records: []
        };
    }
}

/**
 * Get Aadhaar Monthly Enrolment Data
 * Fields: date, state, district, pincode, age_0_5, age_5_17, age_18_greater
 */
export async function getEnrolmentData(options = {}) {
    return fetchData(RESOURCES.ENROLMENT, options);
}

/**
 * Get Aadhaar Demographic Monthly Update Data
 * Fields: date, state, district, pincode, demo_age_5_17, demo_age_17_
 */
export async function getDemographicData(options = {}) {
    return fetchData(RESOURCES.DEMOGRAPHIC, options);
}

/**
 * Get Aadhaar Biometric Monthly Update Data
 * Fields: date, state, district, pincode, bio_age_5_17, bio_age_17_
 */
export async function getBiometricData(options = {}) {
    return fetchData(RESOURCES.BIOMETRIC, options);
}

/**
 * Get all data sources combined
 */
export async function getAllData(options = {}) {
    const [enrolment, demographic, biometric] = await Promise.all([
        getEnrolmentData(options),
        getDemographicData(options),
        getBiometricData(options)
    ]);

    return { enrolment, demographic, biometric };
}

export default {
    getEnrolmentData,
    getDemographicData,
    getBiometricData,
    getAllData,
    RESOURCES
};
