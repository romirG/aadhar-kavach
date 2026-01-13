// Mock data for UIDAI Aadhaar Analytics Platform
// Realistic Indian state and district data with plausible metrics

export interface StateData {
  id: string;
  name: string;
  code: string;
  enrollmentCoverage: number;
  totalPopulation: number;
  enrolledPopulation: number;
  maleEnrolled: number;
  femaleEnrolled: number;
  genderGap: number;
  activeCenters: number;
  pendingUpdates: number;
  riskScore: number;
  region: 'north' | 'south' | 'east' | 'west' | 'central' | 'northeast';
}

export interface DistrictData {
  id: string;
  name: string;
  stateId: string;
  stateName: string;
  enrollmentCoverage: number;
  population: number;
  enrolledPopulation: number;
  riskScore: number;
  isHotspot: boolean;
}

export interface EnrollmentTrend {
  month: string;
  enrollments: number;
  updates: number;
  forecast?: number;
  forecastLower?: number;
  forecastUpper?: number;
}

export interface AnomalyAlert {
  id: string;
  type: 'high_rejection' | 'duplicate_attempt' | 'operator_anomaly' | 'biometric_failure';
  severity: 'low' | 'medium' | 'high' | 'critical';
  riskScore: number;
  location: string;
  state: string;
  description: string;
  timestamp: string;
  operatorId?: string;
  affectedCount: number;
}

export interface BiometricRisk {
  id: string;
  ageGroup: string;
  state: string;
  failureProbability: number;
  totalAttempts: number;
  failedAttempts: number;
  commonIssue: string;
}

export interface MigrationFlow {
  source: string;
  target: string;
  count: number;
}

// State data with realistic metrics
export const statesData: StateData[] = [
  { id: 'UP', name: 'Uttar Pradesh', code: 'UP', enrollmentCoverage: 94.2, totalPopulation: 23150000, enrolledPopulation: 21807300, maleEnrolled: 11449833, femaleEnrolled: 10357467, genderGap: 5.0, activeCenters: 2840, pendingUpdates: 156000, riskScore: 1.8, region: 'north' },
  { id: 'MH', name: 'Maharashtra', code: 'MH', enrollmentCoverage: 97.8, totalPopulation: 12500000, enrolledPopulation: 12225000, maleEnrolled: 6234750, femaleEnrolled: 5990250, genderGap: 2.0, activeCenters: 1950, pendingUpdates: 45000, riskScore: 1.2, region: 'west' },
  { id: 'BR', name: 'Bihar', code: 'BR', enrollmentCoverage: 88.5, totalPopulation: 12400000, enrolledPopulation: 10974000, maleEnrolled: 6036700, femaleEnrolled: 4937300, genderGap: 10.0, activeCenters: 1200, pendingUpdates: 234000, riskScore: 2.8, region: 'east' },
  { id: 'WB', name: 'West Bengal', code: 'WB', enrollmentCoverage: 92.1, totalPopulation: 9900000, enrolledPopulation: 9117900, maleEnrolled: 4695219, femaleEnrolled: 4422681, genderGap: 3.0, activeCenters: 1450, pendingUpdates: 89000, riskScore: 1.5, region: 'east' },
  { id: 'MP', name: 'Madhya Pradesh', code: 'MP', enrollmentCoverage: 91.3, totalPopulation: 8500000, enrolledPopulation: 7760500, maleEnrolled: 4114665, femaleEnrolled: 3645835, genderGap: 6.0, activeCenters: 1100, pendingUpdates: 112000, riskScore: 2.1, region: 'central' },
  { id: 'RJ', name: 'Rajasthan', code: 'RJ', enrollmentCoverage: 89.7, totalPopulation: 7900000, enrolledPopulation: 7086300, maleEnrolled: 3898465, femaleEnrolled: 3187835, genderGap: 10.0, activeCenters: 980, pendingUpdates: 145000, riskScore: 2.4, region: 'north' },
  { id: 'TN', name: 'Tamil Nadu', code: 'TN', enrollmentCoverage: 98.2, totalPopulation: 7800000, enrolledPopulation: 7659600, maleEnrolled: 3829800, femaleEnrolled: 3829800, genderGap: 0.0, activeCenters: 1680, pendingUpdates: 23000, riskScore: 0.9, region: 'south' },
  { id: 'KA', name: 'Karnataka', code: 'KA', enrollmentCoverage: 96.5, totalPopulation: 6700000, enrolledPopulation: 6465500, maleEnrolled: 3329235, femaleEnrolled: 3136265, genderGap: 3.0, activeCenters: 1340, pendingUpdates: 34000, riskScore: 1.1, region: 'south' },
  { id: 'GJ', name: 'Gujarat', code: 'GJ', enrollmentCoverage: 95.8, totalPopulation: 6500000, enrolledPopulation: 6227000, maleEnrolled: 3237040, femaleEnrolled: 2989960, genderGap: 4.0, activeCenters: 1120, pendingUpdates: 56000, riskScore: 1.3, region: 'west' },
  { id: 'AP', name: 'Andhra Pradesh', code: 'AP', enrollmentCoverage: 97.1, totalPopulation: 5300000, enrolledPopulation: 5146300, maleEnrolled: 2573150, femaleEnrolled: 2573150, genderGap: 0.0, activeCenters: 980, pendingUpdates: 28000, riskScore: 1.0, region: 'south' },
  { id: 'OR', name: 'Odisha', code: 'OR', enrollmentCoverage: 90.4, totalPopulation: 4600000, enrolledPopulation: 4158400, maleEnrolled: 2162368, femaleEnrolled: 1996032, genderGap: 4.0, activeCenters: 720, pendingUpdates: 98000, riskScore: 1.9, region: 'east' },
  { id: 'TG', name: 'Telangana', code: 'TG', enrollmentCoverage: 96.8, totalPopulation: 3800000, enrolledPopulation: 3678400, maleEnrolled: 1857384, femaleEnrolled: 1821016, genderGap: 1.0, activeCenters: 890, pendingUpdates: 19000, riskScore: 1.0, region: 'south' },
  { id: 'KL', name: 'Kerala', code: 'KL', enrollmentCoverage: 99.1, totalPopulation: 3500000, enrolledPopulation: 3468500, maleEnrolled: 1699165, femaleEnrolled: 1769335, genderGap: -2.0, activeCenters: 780, pendingUpdates: 8000, riskScore: 0.7, region: 'south' },
  { id: 'JH', name: 'Jharkhand', code: 'JH', enrollmentCoverage: 87.2, totalPopulation: 3700000, enrolledPopulation: 3226400, maleEnrolled: 1742256, femaleEnrolled: 1484144, genderGap: 8.0, activeCenters: 540, pendingUpdates: 156000, riskScore: 2.6, region: 'east' },
  { id: 'AS', name: 'Assam', code: 'AS', enrollmentCoverage: 85.6, totalPopulation: 3500000, enrolledPopulation: 2996000, maleEnrolled: 1557920, femaleEnrolled: 1438080, genderGap: 4.0, activeCenters: 480, pendingUpdates: 178000, riskScore: 2.9, region: 'northeast' },
  { id: 'PB', name: 'Punjab', code: 'PB', enrollmentCoverage: 94.5, totalPopulation: 3000000, enrolledPopulation: 2835000, maleEnrolled: 1474200, femaleEnrolled: 1360800, genderGap: 4.0, activeCenters: 620, pendingUpdates: 42000, riskScore: 1.4, region: 'north' },
  { id: 'HR', name: 'Haryana', code: 'HR', enrollmentCoverage: 93.2, totalPopulation: 2800000, enrolledPopulation: 2609600, maleEnrolled: 1409184, femaleEnrolled: 1200416, genderGap: 8.0, activeCenters: 540, pendingUpdates: 67000, riskScore: 1.7, region: 'north' },
  { id: 'CG', name: 'Chhattisgarh', code: 'CG', enrollmentCoverage: 89.8, totalPopulation: 2900000, enrolledPopulation: 2604200, maleEnrolled: 1380226, femaleEnrolled: 1223974, genderGap: 6.0, activeCenters: 420, pendingUpdates: 89000, riskScore: 2.2, region: 'central' },
  { id: 'JK', name: 'Jammu & Kashmir', code: 'JK', enrollmentCoverage: 82.4, totalPopulation: 1400000, enrolledPopulation: 1153600, maleEnrolled: 611408, femaleEnrolled: 542192, genderGap: 6.0, activeCenters: 280, pendingUpdates: 134000, riskScore: 3.1, region: 'north' },
  { id: 'UK', name: 'Uttarakhand', code: 'UK', enrollmentCoverage: 91.6, totalPopulation: 1100000, enrolledPopulation: 1007600, maleEnrolled: 533028, femaleEnrolled: 474572, genderGap: 6.0, activeCenters: 320, pendingUpdates: 34000, riskScore: 1.6, region: 'north' },
  { id: 'HP', name: 'Himachal Pradesh', code: 'HP', enrollmentCoverage: 95.4, totalPopulation: 750000, enrolledPopulation: 715500, maleEnrolled: 364605, femaleEnrolled: 350895, genderGap: 2.0, activeCenters: 240, pendingUpdates: 12000, riskScore: 1.1, region: 'north' },
  { id: 'TR', name: 'Tripura', code: 'TR', enrollmentCoverage: 88.9, totalPopulation: 400000, enrolledPopulation: 355600, maleEnrolled: 184912, femaleEnrolled: 170688, genderGap: 4.0, activeCenters: 120, pendingUpdates: 23000, riskScore: 2.0, region: 'northeast' },
  { id: 'ML', name: 'Meghalaya', code: 'ML', enrollmentCoverage: 79.3, totalPopulation: 350000, enrolledPopulation: 277550, maleEnrolled: 138775, femaleEnrolled: 138775, genderGap: 0.0, activeCenters: 85, pendingUpdates: 45000, riskScore: 3.2, region: 'northeast' },
  { id: 'MN', name: 'Manipur', code: 'MN', enrollmentCoverage: 81.2, totalPopulation: 310000, enrolledPopulation: 251720, maleEnrolled: 128378, femaleEnrolled: 123342, genderGap: 2.0, activeCenters: 78, pendingUpdates: 38000, riskScore: 3.0, region: 'northeast' },
  { id: 'NL', name: 'Nagaland', code: 'NL', enrollmentCoverage: 76.8, totalPopulation: 220000, enrolledPopulation: 168960, maleEnrolled: 86330, femaleEnrolled: 82630, genderGap: 2.0, activeCenters: 56, pendingUpdates: 34000, riskScore: 3.4, region: 'northeast' },
  { id: 'GA', name: 'Goa', code: 'GA', enrollmentCoverage: 98.5, totalPopulation: 160000, enrolledPopulation: 157600, maleEnrolled: 80376, femaleEnrolled: 77224, genderGap: 2.0, activeCenters: 65, pendingUpdates: 2000, riskScore: 0.8, region: 'west' },
  { id: 'AR', name: 'Arunachal Pradesh', code: 'AR', enrollmentCoverage: 72.1, totalPopulation: 150000, enrolledPopulation: 108150, maleEnrolled: 56238, femaleEnrolled: 51912, genderGap: 4.0, activeCenters: 42, pendingUpdates: 28000, riskScore: 3.6, region: 'northeast' },
  { id: 'MZ', name: 'Mizoram', code: 'MZ', enrollmentCoverage: 84.6, totalPopulation: 120000, enrolledPopulation: 101520, maleEnrolled: 50760, femaleEnrolled: 50760, genderGap: 0.0, activeCenters: 38, pendingUpdates: 12000, riskScore: 2.3, region: 'northeast' },
  { id: 'SK', name: 'Sikkim', code: 'SK', enrollmentCoverage: 93.8, totalPopulation: 70000, enrolledPopulation: 65660, maleEnrolled: 34143, femaleEnrolled: 31517, genderGap: 4.0, activeCenters: 28, pendingUpdates: 3000, riskScore: 1.2, region: 'northeast' },
  { id: 'DL', name: 'Delhi', code: 'DL', enrollmentCoverage: 96.2, totalPopulation: 1900000, enrolledPopulation: 1827800, maleEnrolled: 969334, femaleEnrolled: 858466, genderGap: 6.0, activeCenters: 890, pendingUpdates: 45000, riskScore: 1.3, region: 'north' },
];

// District-level data for hotspot analysis
export const districtsData: DistrictData[] = [
  // Uttar Pradesh districts
  { id: 'UP-LKO', name: 'Lucknow', stateId: 'UP', stateName: 'Uttar Pradesh', enrollmentCoverage: 96.5, population: 4590000, enrolledPopulation: 4429350, riskScore: 1.2, isHotspot: false },
  { id: 'UP-VNS', name: 'Varanasi', stateId: 'UP', stateName: 'Uttar Pradesh', enrollmentCoverage: 92.3, population: 3680000, enrolledPopulation: 3396640, riskScore: 1.8, isHotspot: false },
  { id: 'UP-AGR', name: 'Agra', stateId: 'UP', stateName: 'Uttar Pradesh', enrollmentCoverage: 91.8, population: 4420000, enrolledPopulation: 4057560, riskScore: 2.0, isHotspot: false },
  { id: 'UP-ALD', name: 'Prayagraj', stateId: 'UP', stateName: 'Uttar Pradesh', enrollmentCoverage: 89.2, population: 5960000, enrolledPopulation: 5316320, riskScore: 2.4, isHotspot: true },
  { id: 'UP-KNP', name: 'Kanpur', stateId: 'UP', stateName: 'Uttar Pradesh', enrollmentCoverage: 93.1, population: 4580000, enrolledPopulation: 4263980, riskScore: 1.6, isHotspot: false },
  { id: 'UP-GZB', name: 'Ghaziabad', stateId: 'UP', stateName: 'Uttar Pradesh', enrollmentCoverage: 97.2, population: 4680000, enrolledPopulation: 4548960, riskScore: 1.1, isHotspot: false },
  { id: 'UP-MRT', name: 'Meerut', stateId: 'UP', stateName: 'Uttar Pradesh', enrollmentCoverage: 94.5, population: 3440000, enrolledPopulation: 3250800, riskScore: 1.4, isHotspot: false },
  { id: 'UP-BHR', name: 'Bahraich', stateId: 'UP', stateName: 'Uttar Pradesh', enrollmentCoverage: 78.4, population: 3480000, enrolledPopulation: 2728320, riskScore: 3.8, isHotspot: true },
  { id: 'UP-SRJ', name: 'Shravasti', stateId: 'UP', stateName: 'Uttar Pradesh', enrollmentCoverage: 72.1, population: 1180000, enrolledPopulation: 850780, riskScore: 4.2, isHotspot: true },
  { id: 'UP-BLR', name: 'Balrampur', stateId: 'UP', stateName: 'Uttar Pradesh', enrollmentCoverage: 75.6, population: 2150000, enrolledPopulation: 1625400, riskScore: 3.9, isHotspot: true },
  
  // Bihar districts
  { id: 'BR-PAT', name: 'Patna', stateId: 'BR', stateName: 'Bihar', enrollmentCoverage: 94.2, population: 5770000, enrolledPopulation: 5435340, riskScore: 1.5, isHotspot: false },
  { id: 'BR-GAY', name: 'Gaya', stateId: 'BR', stateName: 'Bihar', enrollmentCoverage: 86.3, population: 4390000, enrolledPopulation: 3788570, riskScore: 2.8, isHotspot: true },
  { id: 'BR-PUR', name: 'Purnia', stateId: 'BR', stateName: 'Bihar', enrollmentCoverage: 81.2, population: 3280000, enrolledPopulation: 2663360, riskScore: 3.4, isHotspot: true },
  { id: 'BR-KIS', name: 'Kishanganj', stateId: 'BR', stateName: 'Bihar', enrollmentCoverage: 74.8, population: 1690000, enrolledPopulation: 1264120, riskScore: 4.1, isHotspot: true },
  { id: 'BR-ARR', name: 'Araria', stateId: 'BR', stateName: 'Bihar', enrollmentCoverage: 76.5, population: 2810000, enrolledPopulation: 2149650, riskScore: 3.8, isHotspot: true },
  
  // Maharashtra districts
  { id: 'MH-MUM', name: 'Mumbai', stateId: 'MH', stateName: 'Maharashtra', enrollmentCoverage: 98.9, population: 12440000, enrolledPopulation: 12303160, riskScore: 0.8, isHotspot: false },
  { id: 'MH-PUN', name: 'Pune', stateId: 'MH', stateName: 'Maharashtra', enrollmentCoverage: 98.2, population: 9430000, enrolledPopulation: 9260260, riskScore: 0.9, isHotspot: false },
  { id: 'MH-NGB', name: 'Nagpur', stateId: 'MH', stateName: 'Maharashtra', enrollmentCoverage: 97.5, population: 4650000, enrolledPopulation: 4533750, riskScore: 1.1, isHotspot: false },
  { id: 'MH-GAD', name: 'Gadchiroli', stateId: 'MH', stateName: 'Maharashtra', enrollmentCoverage: 84.6, population: 1070000, enrolledPopulation: 905220, riskScore: 3.2, isHotspot: true },
  { id: 'MH-NAN', name: 'Nandurbar', stateId: 'MH', stateName: 'Maharashtra', enrollmentCoverage: 86.2, population: 1650000, enrolledPopulation: 1422300, riskScore: 2.9, isHotspot: true },
  
  // Rajasthan districts
  { id: 'RJ-JAI', name: 'Jaipur', stateId: 'RJ', stateName: 'Rajasthan', enrollmentCoverage: 95.8, population: 6620000, enrolledPopulation: 6345960, riskScore: 1.2, isHotspot: false },
  { id: 'RJ-JDH', name: 'Jodhpur', stateId: 'RJ', stateName: 'Rajasthan', enrollmentCoverage: 91.2, population: 3690000, enrolledPopulation: 3365280, riskScore: 1.9, isHotspot: false },
  { id: 'RJ-BSW', name: 'Banswara', stateId: 'RJ', stateName: 'Rajasthan', enrollmentCoverage: 78.4, population: 1800000, enrolledPopulation: 1411200, riskScore: 3.6, isHotspot: true },
  { id: 'RJ-DNG', name: 'Dungarpur', stateId: 'RJ', stateName: 'Rajasthan', enrollmentCoverage: 76.9, population: 1390000, enrolledPopulation: 1068910, riskScore: 3.8, isHotspot: true },
  { id: 'RJ-JLR', name: 'Jaisalmer', stateId: 'RJ', stateName: 'Rajasthan', enrollmentCoverage: 82.3, population: 670000, enrolledPopulation: 551410, riskScore: 3.2, isHotspot: true },
  
  // Additional districts for other states
  { id: 'TN-CHE', name: 'Chennai', stateId: 'TN', stateName: 'Tamil Nadu', enrollmentCoverage: 99.2, population: 4640000, enrolledPopulation: 4602880, riskScore: 0.6, isHotspot: false },
  { id: 'KA-BLR', name: 'Bengaluru', stateId: 'KA', stateName: 'Karnataka', enrollmentCoverage: 98.1, population: 8440000, enrolledPopulation: 8279640, riskScore: 0.8, isHotspot: false },
  { id: 'GJ-AMD', name: 'Ahmedabad', stateId: 'GJ', stateName: 'Gujarat', enrollmentCoverage: 97.6, population: 5570000, enrolledPopulation: 5436320, riskScore: 1.0, isHotspot: false },
  { id: 'AS-KAM', name: 'Kamrup', stateId: 'AS', stateName: 'Assam', enrollmentCoverage: 89.4, population: 1520000, enrolledPopulation: 1358880, riskScore: 2.4, isHotspot: false },
  { id: 'AS-DHU', name: 'Dhubri', stateId: 'AS', stateName: 'Assam', enrollmentCoverage: 72.8, population: 1950000, enrolledPopulation: 1419600, riskScore: 4.3, isHotspot: true },
  { id: 'JK-SRN', name: 'Srinagar', stateId: 'JK', stateName: 'Jammu & Kashmir', enrollmentCoverage: 86.5, population: 1270000, enrolledPopulation: 1098550, riskScore: 2.8, isHotspot: false },
  { id: 'JK-KUP', name: 'Kupwara', stateId: 'JK', stateName: 'Jammu & Kashmir', enrollmentCoverage: 71.2, population: 870000, enrolledPopulation: 619440, riskScore: 4.5, isHotspot: true },
];

// Historical enrollment trends and forecast data
export const enrollmentTrends: EnrollmentTrend[] = [
  { month: 'Jan 2024', enrollments: 1234000, updates: 890000 },
  { month: 'Feb 2024', enrollments: 1156000, updates: 912000 },
  { month: 'Mar 2024', enrollments: 1389000, updates: 1023000 },
  { month: 'Apr 2024', enrollments: 1278000, updates: 956000 },
  { month: 'May 2024', enrollments: 1456000, updates: 1089000 },
  { month: 'Jun 2024', enrollments: 1345000, updates: 1145000 },
  { month: 'Jul 2024', enrollments: 1267000, updates: 1067000 },
  { month: 'Aug 2024', enrollments: 1523000, updates: 1234000 },
  { month: 'Sep 2024', enrollments: 1478000, updates: 1189000 },
  { month: 'Oct 2024', enrollments: 1612000, updates: 1345000 },
  { month: 'Nov 2024', enrollments: 1534000, updates: 1278000 },
  { month: 'Dec 2024', enrollments: 1389000, updates: 1156000 },
  // Forecast data (future months)
  { month: 'Jan 2025', enrollments: 1420000, updates: 1200000, forecast: 1450000, forecastLower: 1380000, forecastUpper: 1520000 },
  { month: 'Feb 2025', enrollments: 1380000, updates: 1150000, forecast: 1410000, forecastLower: 1330000, forecastUpper: 1490000 },
  { month: 'Mar 2025', enrollments: 1510000, updates: 1280000, forecast: 1540000, forecastLower: 1450000, forecastUpper: 1630000 },
  { month: 'Apr 2025', enrollments: 1450000, updates: 1220000, forecast: 1480000, forecastLower: 1390000, forecastUpper: 1570000 },
  { month: 'May 2025', enrollments: 1580000, updates: 1350000, forecast: 1620000, forecastLower: 1520000, forecastUpper: 1720000 },
  { month: 'Jun 2025', enrollments: 1520000, updates: 1300000, forecast: 1560000, forecastLower: 1460000, forecastUpper: 1660000 },
];

// Anomaly alerts for fraud detection
export const anomalyAlerts: AnomalyAlert[] = [
  {
    id: 'ALT-001',
    type: 'high_rejection',
    severity: 'critical',
    riskScore: 4.2,
    location: 'Kishanganj District Center',
    state: 'Bihar',
    description: 'Rejection rate 340% above normal. 89 rejections in last 24 hours vs average of 20.',
    timestamp: '2025-01-13T08:45:00',
    operatorId: 'OP-BR-4521',
    affectedCount: 89
  },
  {
    id: 'ALT-002',
    type: 'duplicate_attempt',
    severity: 'high',
    riskScore: 3.8,
    location: 'Shravasti Enrollment Center',
    state: 'Uttar Pradesh',
    description: 'Multiple duplicate enrollment attempts detected. Same biometrics submitted 12 times.',
    timestamp: '2025-01-13T07:30:00',
    operatorId: 'OP-UP-7892',
    affectedCount: 12
  },
  {
    id: 'ALT-003',
    type: 'operator_anomaly',
    severity: 'high',
    riskScore: 3.5,
    location: 'Dhubri Processing Center',
    state: 'Assam',
    description: 'Operator processing 4x normal volume. 156 enrollments in 2 hours.',
    timestamp: '2025-01-13T06:15:00',
    operatorId: 'OP-AS-3456',
    affectedCount: 156
  },
  {
    id: 'ALT-004',
    type: 'biometric_failure',
    severity: 'medium',
    riskScore: 2.8,
    location: 'Kupwara Mobile Unit',
    state: 'Jammu & Kashmir',
    description: 'Fingerprint capture failure rate at 28%. Equipment calibration may be needed.',
    timestamp: '2025-01-13T05:00:00',
    affectedCount: 34
  },
  {
    id: 'ALT-005',
    type: 'high_rejection',
    severity: 'medium',
    riskScore: 2.6,
    location: 'Banswara Tehsil Office',
    state: 'Rajasthan',
    description: 'Photo quality rejections increased by 180%. Lighting issues suspected.',
    timestamp: '2025-01-13T04:30:00',
    affectedCount: 45
  },
  {
    id: 'ALT-006',
    type: 'duplicate_attempt',
    severity: 'medium',
    riskScore: 2.4,
    location: 'Purnia Block Center',
    state: 'Bihar',
    description: 'Unusual pattern of re-enrollment requests from same household addresses.',
    timestamp: '2025-01-12T22:00:00',
    affectedCount: 8
  },
  {
    id: 'ALT-007',
    type: 'operator_anomaly',
    severity: 'low',
    riskScore: 1.9,
    location: 'Gaya District Center',
    state: 'Bihar',
    description: 'Processing time 50% faster than average. Review for quality check.',
    timestamp: '2025-01-12T20:15:00',
    operatorId: 'OP-BR-2234',
    affectedCount: 78
  },
  {
    id: 'ALT-008',
    type: 'biometric_failure',
    severity: 'low',
    riskScore: 1.5,
    location: 'Gadchiroli Mobile Camp',
    state: 'Maharashtra',
    description: 'Iris scan failures for elderly population above threshold.',
    timestamp: '2025-01-12T18:45:00',
    affectedCount: 23
  },
];

// Biometric risk data
export const biometricRisks: BiometricRisk[] = [
  { id: 'BR-001', ageGroup: '70+', state: 'Bihar', failureProbability: 0.34, totalAttempts: 1250, failedAttempts: 425, commonIssue: 'Fingerprint wear' },
  { id: 'BR-002', ageGroup: '70+', state: 'Uttar Pradesh', failureProbability: 0.31, totalAttempts: 2340, failedAttempts: 725, commonIssue: 'Fingerprint wear' },
  { id: 'BR-003', ageGroup: '60-70', state: 'Rajasthan', failureProbability: 0.28, totalAttempts: 1890, failedAttempts: 529, commonIssue: 'Dry skin' },
  { id: 'BR-004', ageGroup: '70+', state: 'Jharkhand', failureProbability: 0.32, totalAttempts: 980, failedAttempts: 314, commonIssue: 'Manual labor damage' },
  { id: 'BR-005', ageGroup: '60-70', state: 'Madhya Pradesh', failureProbability: 0.25, totalAttempts: 1560, failedAttempts: 390, commonIssue: 'Fingerprint wear' },
  { id: 'BR-006', ageGroup: '70+', state: 'Assam', failureProbability: 0.29, totalAttempts: 670, failedAttempts: 194, commonIssue: 'Iris opacity' },
  { id: 'BR-007', ageGroup: '0-5', state: 'Bihar', failureProbability: 0.22, totalAttempts: 3450, failedAttempts: 759, commonIssue: 'Underdeveloped biometrics' },
  { id: 'BR-008', ageGroup: '0-5', state: 'Uttar Pradesh', failureProbability: 0.21, totalAttempts: 5670, failedAttempts: 1191, commonIssue: 'Underdeveloped biometrics' },
];

// Migration flow data
export const migrationFlows: MigrationFlow[] = [
  { source: 'Bihar', target: 'Delhi', count: 234000 },
  { source: 'Uttar Pradesh', target: 'Maharashtra', count: 312000 },
  { source: 'Bihar', target: 'Maharashtra', count: 189000 },
  { source: 'Rajasthan', target: 'Gujarat', count: 156000 },
  { source: 'Odisha', target: 'Gujarat', count: 134000 },
  { source: 'Jharkhand', target: 'West Bengal', count: 112000 },
  { source: 'Uttar Pradesh', target: 'Gujarat', count: 98000 },
  { source: 'West Bengal', target: 'Karnataka', count: 87000 },
  { source: 'Madhya Pradesh', target: 'Maharashtra', count: 78000 },
  { source: 'Chhattisgarh', target: 'Maharashtra', count: 65000 },
];

// Helper functions
export const getStateById = (id: string): StateData | undefined => {
  return statesData.find(state => state.id === id);
};

export const getDistrictsByState = (stateId: string): DistrictData[] => {
  return districtsData.filter(district => district.stateId === stateId);
};

export const getHotspotDistricts = (): DistrictData[] => {
  return districtsData.filter(district => district.isHotspot);
};

export const getTotalEnrollments = (): number => {
  return statesData.reduce((sum, state) => sum + state.enrolledPopulation, 0);
};

export const getTotalPendingUpdates = (): number => {
  return statesData.reduce((sum, state) => sum + state.pendingUpdates, 0);
};

export const getAverageEnrollmentCoverage = (): number => {
  const total = statesData.reduce((sum, state) => sum + state.enrollmentCoverage, 0);
  return total / statesData.length;
};

export const getActiveAlertsCount = (): number => {
  return anomalyAlerts.filter(alert => alert.severity === 'critical' || alert.severity === 'high').length;
};

export const formatNumber = (num: number): string => {
  if (num >= 10000000) return (num / 10000000).toFixed(2) + ' Cr';
  if (num >= 100000) return (num / 100000).toFixed(2) + ' L';
  if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
  return num.toString();
};

export const formatPercentage = (num: number): string => {
  return num.toFixed(1) + '%';
};
