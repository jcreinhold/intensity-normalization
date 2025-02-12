"""CLI for calculating tissue membership image."""

__all__ = ["tissue_membership_main", "tissue_membership_parser"]

from intensity_normalization.util.tissue_membership import TissueMembershipFinder

# main functions and parsers for CLI
tissue_membership_parser = TissueMembershipFinder.parser()
tissue_membership_main = TissueMembershipFinder.main(tissue_membership_parser)
