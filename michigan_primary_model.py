import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from scipy import stats


@dataclass
class CountyBaseline:
    """County-level baseline expectations."""
    name: str
    margin: float  # El-Sayed margin in percentage points
    turnout: int  # Projected turnout for 1.42M statewide
    is_dem_heavy: bool = True


class MichiganPrimaryModel:
    """
    Live Bayesian election-night model for Michigan Democratic Primary (El-Sayed vs Stevens).
    
    Features:
    - County-level baseline expectations from polling/prior elections
    - Early vote vs Election Day vote mode-specific modeling
    - Bayesian updating based on reported results
    - DerSimonian-Laird meta-analysis for uncertainty estimation
    - Monte Carlo simulation for final margin and win probability
    """
    
    def __init__(self, counties_data: pd.DataFrame):
        """
        Initialize model with county baselines.
        
        Args:
            counties_data: DataFrame with columns ['county', 'margin', 'turnout']
                margin: El-Sayed margin (positive = El-Sayed, negative = Stevens)
                turnout: Projected total votes for that county at 1.42M statewide
        """
        self.counties = counties_data.copy()
        self.counties['margin'] = self.counties['margin'].astype(float)
        self.counties['turnout'] = self.counties['turnout'].astype(int)
        
        # Model parameters
        self.early_vote_advantage = 0.20  # Stevens performs 20% better in early/mail votes
        self.n_simulations = 20000
        self.credibility_exponent = 2.0
        self.outlier_lambda = 3.0
        self.tau_floor = 0.08
        
        # State-level turnout
        self.total_turnout = counties_data['turnout'].sum()
        
        # Results tracking
        self.reported_votes = {}  # {county: {'el_sayed': int, 'stevens': int}}
        self.vote_mode = {}  # {county: 'early'|'election_day'|'mixed'}
        
    def set_reported_votes(self, county: str, el_sayed: int, stevens: int, 
                          mode: str = 'mixed') -> None:
        """
        Record votes reported for a county.
        
        Args:
            county: County name
            el_sayed: El-Sayed votes reported
            stevens: Stevens votes reported
            mode: 'early' (mail/early in-person), 'election_day', or 'mixed'
        """
        self.reported_votes[county] = {
            'el_sayed': el_sayed,
            'stevens': stevens
        }
        self.vote_mode[county] = mode
    
    def get_county_data(self, county: str) -> Optional[pd.Series]:
        """Get baseline data for a county."""
        matches = self.counties[self.counties['county'].str.lower() == county.lower()]
        return matches.iloc[0] if len(matches) > 0 else None
    
    def _get_early_vote_adjusted_baseline(self, county_margin: float, 
                                         mode: str = 'mixed') -> float:
        """
        Adjust baseline margin based on vote mode.
        
        Stevens performs 20% better in early/mail votes.
        Example: if baseline is El-Sayed +10, early votes expected to be ~El-Sayed +8
        """
        if mode == 'early':
            # Stevens shifts baseline toward him by 20% of the gap
            return county_margin * (1 - self.early_vote_advantage)
        elif mode == 'election_day':
            # Election day votes move further toward El-Sayed
            return county_margin * (1 + self.early_vote_advantage * 0.5)
        else:  # mixed
            return county_margin
    
    def _calculate_county_shift(self, county: str) -> Tuple[float, float]:
        """
        Calculate observed margin shift from baseline for a county with reported votes.
        
        Returns:
            (observed_margin, observed_margin_se)
        """
        if county not in self.reported_votes:
            return None, None
        
        county_data = self.get_county_data(county)
        if county_data is None:
            return None, None
        
        reported = self.reported_votes[county]
        mode = self.vote_mode.get(county, 'mixed')
        
        votes = reported['el_sayed'] + reported['stevens']
        if votes == 0:
            return None, None
        
        # Observed margin
        el_sayed_pct = reported['el_sayed'] / votes
        stevens_pct = reported['stevens'] / votes
        observed_margin = (el_sayed_pct - stevens_pct) * 100
        
        # Baseline adjusted for vote mode
        baseline_margin = self._get_early_vote_adjusted_baseline(
            county_data['margin'], mode
        )
        
        # Shift from baseline
        shift = observed_margin - baseline_margin
        
        # Standard error of observed margin
        se_margin = 100 * np.sqrt(el_sayed_pct * stevens_pct / votes)
        
        return shift, se_margin
    
    def _dersimonian_laird_meta_analysis(self) -> Tuple[float, float]:
        """
        DerSimonian-Laird random effects meta-analysis of county shifts.
        
        Returns:
            (pooled_shift, heterogeneity_tau)
        """
        shifts = []
        ses = []
        
        for county in self.reported_votes.keys():
            shift, se = self._calculate_county_shift(county)
            if shift is not None and se is not None and se > 0:
                shifts.append(shift)
                ses.append(se)
        
        if not shifts:
            return 0.0, self.tau_floor
        
        shifts = np.array(shifts)
        ses = np.array(ses)
        
        # Fixed effects estimate and heterogeneity test
        weights = 1.0 / (ses ** 2)
        pooled_fe = np.sum(shifts * weights) / np.sum(weights)
        q_stat = np.sum(weights * (shifts - pooled_fe) ** 2)
        
        k = len(shifts)
        c = np.sum(weights) - np.sum(weights ** 2) / np.sum(weights)
        
        # Estimate tau (between-study heterogeneity)
        if q_stat <= (k - 1):
            tau = self.tau_floor
        else:
            tau = np.sqrt((q_stat - (k - 1)) / c)
            tau = max(tau, self.tau_floor)
        
        # Random effects estimate
        weights_re = 1.0 / (ses ** 2 + tau ** 2)
        pooled_re = np.sum(shifts * weights_re) / np.sum(weights_re)
        se_re = np.sqrt(1.0 / np.sum(weights_re))
        
        return pooled_re, max(tau, se_re)
    
    def _get_unreported_county_baseline_shift(self, reported_shift: float, 
                                              n_reported: int) -> float:
        """
        Credibility-weighted shift for unreported counties.
        
        Applies credibility interval to dampen shifts based on how many
        counties have reported.
        """
        n_total = len(self.counties)
        credibility = (n_reported / n_total) ** self.credibility_exponent
        
        # Dampen shift for unreported counties
        return reported_shift * credibility
    
    def _project_unreported_votes(self, pooled_shift: float, 
                                  heterogeneity_tau: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project vote margins for all unreported counties.
        
        Returns:
            (el_sayed_margins, stevens_margins) as arrays
        """
        n_reported = len(self.reported_votes)
        credible_shift = self._get_unreported_county_baseline_shift(
            pooled_shift, n_reported
        )
        
        unreported_margins = []
        unreported_se = []
        
        for _, county_row in self.counties.iterrows():
            county_name = county_row['county']
            
            if county_name not in self.reported_votes:
                # Apply credible shift to baseline
                projected_margin = county_row['margin'] + credible_shift
                
                # Uncertainty scales with heterogeneity
                margin_se = heterogeneity_tau + abs(credible_shift) * 0.3
                
                unreported_margins.append(projected_margin)
                unreported_se.append(margin_se)
        
        return np.array(unreported_margins), np.array(unreported_se)
    
    def simulate_election(self) -> Dict:
        """
        Run full election simulation with Monte Carlo.
        
        Returns:
            Dictionary with:
            - 'el_sayed_votes': array of final vote totals across simulations
            - 'stevens_votes': array of final vote totals across simulations
            - 'el_sayed_margin': array of final margins
            - 'el_sayed_win_probability': P(El-Sayed wins)
            - 'median_margin': Median final margin
            - 'margin_ci_lower': 5th percentile margin
            - 'margin_ci_upper': 95th percentile margin
            - 'el_sayed_median_votes': Median El-Sayed vote total
            - 'stevens_median_votes': Median Stevens vote total
        """
        
        # Step 1: Meta-analysis of reported counties
        pooled_shift, tau = self._dersimonian_laird_meta_analysis()
        
        # Step 2: Project unreported counties
        unreported_margins, unreported_se = self._project_unreported_votes(
            pooled_shift, tau
        )
        
        # Step 3: Monte Carlo simulation
        simulations_el_sayed = np.zeros(self.n_simulations)
        simulations_stevens = np.zeros(self.n_simulations)
        
        for sim in range(self.n_simulations):
            total_el_sayed = 0
            total_stevens = 0
            
            # Reported counties: fixed
            for county in self.reported_votes.keys():
                county_votes = self.reported_votes[county]
                total_el_sayed += county_votes['el_sayed']
                total_stevens += county_votes['stevens']
            
            # Unreported counties: sample from baseline with uncertainty
            unreported_idx = 0
            for _, county_row in self.counties.iterrows():
                county_name = county_row['county']
                
                if county_name not in self.reported_votes:
                    # Sample margin from distribution
                    projected_margin = unreported_margins[unreported_idx]
                    margin_se = unreported_se[unreported_idx]
                    
                    sampled_margin = np.random.normal(projected_margin, margin_se)
                    
                    # Convert margin to votes
                    county_turnout = county_row['turnout']
                    
                    # Margin % to vote split
                    el_sayed_pct = (50 + sampled_margin / 2) / 100
                    el_sayed_pct = np.clip(el_sayed_pct, 0.01, 0.99)
                    
                    el_sayed_votes = int(county_turnout * el_sayed_pct)
                    stevens_votes = county_turnout - el_sayed_votes
                    
                    total_el_sayed += el_sayed_votes
                    total_stevens += stevens_votes
                    
                    unreported_idx += 1
            
            simulations_el_sayed[sim] = total_el_sayed
            simulations_stevens[sim] = total_stevens
        
        # Step 4: Calculate results
        margins = ((simulations_el_sayed - simulations_stevens) / 
                   (simulations_el_sayed + simulations_stevens) * 100)
        
        el_sayed_wins = np.sum(simulations_el_sayed > simulations_stevens)
        win_prob = el_sayed_wins / self.n_simulations
        
        return {
            'el_sayed_votes': simulations_el_sayed,
            'stevens_votes': simulations_stevens,
            'el_sayed_margin': margins,
            'el_sayed_win_probability': win_prob,
            'median_margin': np.median(margins),
            'margin_ci_lower': np.percentile(margins, 5),
            'margin_ci_upper': np.percentile(margins, 95),
            'el_sayed_median_votes': np.median(simulations_el_sayed),
            'stevens_median_votes': np.median(simulations_stevens),
            'el_sayed_mean_votes': np.mean(simulations_el_sayed),
            'stevens_mean_votes': np.mean(simulations_stevens),
            'pooled_shift': pooled_shift,
            'heterogeneity_tau': tau,
            'n_reported_counties': len(self.reported_votes),
            'n_total_counties': len(self.counties)
        }
    
    def get_summary(self) -> Dict:
        """
        Quick summary of current state.
        
        Returns:
            Dictionary with vote totals and basic stats
        """
        total_reported_el_sayed = 0
        total_reported_stevens = 0
        
        for county_votes in self.reported_votes.values():
            total_reported_el_sayed += county_votes['el_sayed']
            total_reported_stevens += county_votes['stevens']
        
        reported_total = total_reported_el_sayed + total_reported_stevens
        reported_pct = (reported_total / self.total_turnout * 100) if self.total_turnout > 0 else 0
        
        if reported_total > 0:
            el_sayed_pct = total_reported_el_sayed / reported_total * 100
            stevens_pct = total_reported_stevens / reported_total * 100
        else:
            el_sayed_pct = stevens_pct = 0
        
        return {
            'el_sayed_votes_reported': total_reported_el_sayed,
            'stevens_votes_reported': total_reported_stevens,
            'total_votes_reported': reported_total,
            'pct_votes_reported': reported_pct,
            'el_sayed_pct_reported': el_sayed_pct,
            'stevens_pct_reported': stevens_pct,
            'n_counties_reporting': len(self.reported_votes),
            'n_counties_total': len(self.counties)
        }


def build_michigan_county_data() -> pd.DataFrame:
    """
    Build county baseline data for Michigan Democratic Primary.
    
    Scales Slotkin-Harper 933K election to 1.42M turnout.
    Applies El-Sayed vs Stevens margin baselines.
    
    Returns:
        DataFrame with columns: county, margin, turnout
    """
    
    # Original Slotkin-Harper 933K election turnout by county
    slotkin_harper = {
        'Wayne': (194835, 0.61),
        'Oakland': (152510, 0.77),
        'Macomb': (70816, 0.77),
        'Washtenaw': (62639, 0.80),
        'Kent': (62509, 0.78),
        'Genesee': (43731, 0.69),
        'Ingham': (36376, 0.90),
        'Kalamazoo': (26401, 0.81),
        'Saginaw': (19758, 0.71),
        'Livingston': (16633, 0.95),
        'Muskegon': (16098, 0.74),
        'Eaton': (11208, 0.92),
        'Grand Traverse': (10880, 0.88),
        'Bay': (10388, 0.84),
        'Ottawa': (10259, 0.84),
        'Jackson': (9443, 0.88),
        'St. Clair': (9328, 0.89),
        'Monroe': (9290, 0.86),
        'Berrien': (9261, 0.69),
        'Marquette': (9079, 0.73),
        'Clinton': (7742, 0.95),
        'Calhoun': (7517, 0.77),
        'Allegan': (7113, 0.87),
        'Midland': (6981, 0.87),
        'Lenawee': (5391, 0.83),
        'Shiawassee': (5279, 0.92),
        'Lapeer': (5152, 0.90),
        'Van Buren': (4976, 0.84),
        'Isabella': (4923, 0.84),
        'Leelanau': (4289, 0.91),
        'Montcalm': (3172, 0.85),
        'Emmet': (3080, 0.90),
        'Benzie': (2721, 0.89),
        'Ionia': (2721, 0.89),
        'Alpena': (2691, 0.80),
        'Houghton': (2685, 0.80),
        'Delta': (2596, 0.75),
        'Charlevoix': (2530, 0.90),
        'Tuscola': (2483, 0.83),
        'Manistee': (2381, 0.87),
        'Chippewa': (2154, 0.84),
        'Cass': (2135, 0.75),
        'Mecosta': (1967, 0.85),
        'Iosco': (1958, 0.88),
        'Newaygo': (1945, 0.84),
        'Barry': (1913, 0.85),
        'Antrim': (1895, 0.89),
        'Roscommon': (1889, 0.86),
        'Mason': (1808, 0.88),
        'Cheboygan': (1799, 0.89),
        'Dickinson': (1727, 0.77),
        'Sanilac': (1712, 0.91),
        'Otsego': (1694, 0.89),
        'Clare': (1684, 0.85),
        'Gratiot': (1643, 0.85),
        'Oceana': (1611, 0.89),
        'Wexford': (1562, 0.86),
        'Menominee': (1527, 0.72),
        'Hillsdale': (1523, 0.89),
        'Huron': (1515, 0.87),
        'Gladwin': (1511, 0.84),
        'St. Joseph': (1509, 0.80),
        'Ogemaw': (1405, 0.88),
        'Branch': (1344, 0.86),
        'Presque Isle': (1229, 0.89),
        'Gogebic': (1176, 0.77),
        'Kalkaska': (1136, 0.86),
        'Iron': (1028, 0.79),
        'Alger': (977, 0.82),
        'Osceola': (951, 0.86),
        'Crawford': (903, 0.87),
        'Mackinac': (877, 0.89),
        'Arenac': (824, 0.86),
        'Alcona': (813, 0.85),
        'Lake': (811, 0.68),
        'Ontonagon': (636, 0.78),
        'Missaukee': (605, 0.85),
        'Baraga': (575, 0.70),
        'Schoolcraft': (554, 0.75),
        'Montmorency': (546, 0.85),
        'Oscoda': (391, 0.89),
        'Keweenaw': (277, 0.85),
        'Luce': (240, 0.87),
    }
    
    # El-Sayed vs Stevens baseline margins
    margin_baselines = {
        'Wayne': 8.0, 'Oakland': -8.3, 'Macomb': 12.1, 'Kent': 30.9, 'Washtenaw': 39.4,
        'Genesee': 5.0, 'Ottawa': 41.5, 'Kalamazoo': 21.9, 'Ingham': 53.7, 'Livingston': 51.4,
        'Saginaw': 4.6, 'Muskegon': 12.0, 'St. Clair': 4.3, 'Berrien': 5.2, 'Monroe': 24.5,
        'Jackson': 1.4, 'Allegan': 15.2, 'Calhoun': 10.9, 'Bay': 4.0, 'Eaton': 33.2,
        'Grand Traverse': 37.4, 'Lapeer': 1.1, 'Midland': 33.7, 'Lenawee': 4.5, 'Clinton': 37.9,
        'Marquette': 31.4, 'Shiawassee': 14.5, 'Barry': 11.7, 'Van Buren': 19.2, 'Isabella': 39.5,
        'St. Joseph': -2.2, 'Tuscola': -3.8, 'Montcalm': 8.8, 'Ionia': 16.0, 'Newaygo': 19.5,
        'Cass': 1.1, 'Sanilac': 1.0, 'Delta': 0.0, 'Emmet': 29.9, 'Hillsdale': 3.5, 'Branch': 9.6,
        'Houghton': 35.7, 'Mecosta': 10.0, 'Charlevoix': 38.2, 'Gratiot': -4.0, 'Wexford': 0.1,
        'Chippewa': 1.1, 'Huron': 9.2, 'Antrim': 15.0, 'Mason': 9.9, 'Alpena': 8.3,
        'Cheboygan': 11.1, 'Dickinson': 9.2, 'Leelanau': 66.5, 'Otsego': -6.8, 'Manistee': 20.0,
        'Clare': 4.3, 'Iosco': 8.6, 'Menominee': 2.1, 'Gladwin': 22.8, 'Roscommon': 4.9,
        'Oceana': -0.5, 'Ogemaw': 21.3, 'Osceola': 1.6, 'Benzie': 36.1, 'Kalkaska': 5.4,
        'Arenac': 6.0, 'Missaukee': 3.9, 'Crawford': 17.0, 'Gogebic': 4.4, 'Presque Isle': 8.0,
        'Mackinac': 0.4, 'Iron': 3.7, 'Montmorency': 14.6, 'Alcona': 11.6, 'Lake': -5.9,
        'Alger': 0.1, 'Schoolcraft': 12.8, 'Baraga': 2.3, 'Oscoda': 9.4, 'Ontonagon': -0.3,
        'Luce': -1.6, 'Keweenaw': 5.9
    }
    
    # Scale to 1.42M turnout
    total_baseline = sum(v[0] for v in slotkin_harper.values())
    scaling_factor = 1.42e6 / total_baseline  # ~1.523x
    
    data = []
    for county, (baseline_votes, dem_pct) in slotkin_harper.items():
        scaled_turnout = int(baseline_votes * scaling_factor)
        margin = margin_baselines.get(county, 0.0)
        
        data.append({
            'county': county,
            'margin': margin,
            'turnout': scaled_turnout
        })
    
    return pd.DataFrame(data)


if __name__ == '__main__':
    # Example usage
    print("Building Michigan Democratic Primary model...")
    counties_df = build_michigan_county_data()
    print(f"Loaded {len(counties_df)} counties, {counties_df['turnout'].sum():,} total projected turnout\n")
    
    model = MichiganPrimaryModel(counties_df)
    
    # Example: Report some votes
    print("Example: Recording sample vote reports...")
    model.set_reported_votes('Wayne', el_sayed=30000, stevens=10000, mode='early')
    model.set_reported_votes('Oakland', el_sayed=45000, stevens=15000, mode='mixed')
    model.set_reported_votes('Macomb', el_sayed=25000, stevens=10000, mode='election_day')
    
    summary = model.get_summary()
    print(f"\nVotes Reported: {summary['total_votes_reported']:,} ({summary['pct_votes_reported']:.1f}%)")
    print(f"El-Sayed: {summary['el_sayed_votes_reported']:,} ({summary['el_sayed_pct_reported']:.1f}%)")
    print(f"Stevens: {summary['stevens_votes_reported']:,} ({summary['stevens_pct_reported']:.1f}%)")
    
    print("\nRunning 20,000 simulations...")
    results = model.simulate_election()
    
    print(f"\nProjection Results:")
    print(f"El-Sayed Win Probability: {results['el_sayed_win_probability']*100:.1f}%")
    print(f"Median Final Margin: El-Sayed +{results['median_margin']:.1f}%")
    print(f"90% Confidence Interval: +{results['margin_ci_lower']:.1f}% to +{results['margin_ci_upper']:.1f}%")
    print(f"\nMedian Vote Totals:")
    print(f"El-Sayed: {results['el_sayed_median_votes']:,.0f}")
    print(f"Stevens: {results['stevens_median_votes']:,.0f}")
