#!/usr/bin/perl

use strict;
use warnings;
use Getopt::Long;
# keep if enrichment in one side
# Before: we used unormalised coverage ==> small coverage for small loop & error in controm value (lower value for small loop, so we keep more small than big loops)

##############
# Declaration
##############

my $infile_name ='';
my $outfile = '';

my %Color = ();
$Color{1} = '153,204,255'; #light blue
$Color{2} = '0,0,255'; # blue
my @Header = ();


my @Value = ();
my @Info =  ('Coverage_X','Coverage_X_control','Coverage_Y','Coverage_Y_control','nbX_ref','nbY_ref','nbX_control','nbY_control');
my @Start = ();
my @End = ();
my $ratioi= 0;

# Paramters
my $resolution = 5000;
my $delta_diagonale = 15; # warning, : filter alreadt exisy in derivative ==> 10
my $min_ratio = 2;
my $min_reads_per_bin = 2;
my $round = $resolution *100;
# my $quantile = 0.1;

my $quantile = 0.5;
##############
# Programs
##############

my $result =  GetOptions ("i=s" => \$infile_name, "o=s" => \$outfile, "d=s" => \$delta_diagonale, "f=s" => \$min_ratio, "m=s" => \$min_reads_per_bin, "r=s" => \$resolution);

$outfile = $outfile.'_r'.$resolution.'_d'.$delta_diagonale.'_fc'.$min_ratio.'_m'.$min_reads_per_bin.'.bed';

# my $min_cov_control = $min_reads_per_bin;

open OUT,">".$outfile;
my %CoverageControl = ();
my %CoverageMin = ();
my %CoverageMax = ();
for (my $i_chr=1; $i_chr <= 24; $i_chr ++){
	my $chr = 'chr'.$i_chr;
	@Header = ();
	$chr = 'chrX' if ($i_chr == 23);
	$chr = 'chrY' if ($i_chr == 24);
	my $infile = $infile_name.'_'.$chr.'_enrichment.out';
	if (-e $infile){
		print "read $infile\n";
		# determine min coverage control

		open IN,$infile;
		while (<IN>){
			chomp($_);
			if ($#Header < 0){
				@Header = split(/\t/,$_);
			}
			else{
				my %Column = ();
				readLine($_, \@Header, \%Column);
				# my $size = $Column{'End'} - $Column{'Start'};
				my $size = $Column{'End'} - $Column{'Start'};
				if ($size >= $delta_diagonale*$resolution) {	
					my $size2 = sprintf("%.0f",$size/$round)*$round;
					if (exists($CoverageControl{$size2})){
						# Normalised to take into account different size in the same range
						push(@{$CoverageControl{$size2}},$Column{'Coverage_X_control'}/$size );
						push(@{$CoverageControl{$size2}},$Column{'Coverage_Y_control'}/$size );
					}
					else{
						@{$CoverageControl{$size2}} = ($Column{'Coverage_X_control'}/$size,$Column{'Coverage_Y_control'}/$size);
					}
					# Old version:
					# if (exists($CoverageControl{$size})){
						# push(@{$CoverageControl{$size}},$Column{'Coverage_X_control'} );
						# push(@{$CoverageControl{$size}},$Column{'Coverage_Y_control'} );
					# }
					# else{
						# @{$CoverageControl{$size}} = ($Column{'Coverage_X_control'},$Column{'Coverage_Y_control'});
					# }
				}
			}
		}
		close(IN);
	}
	else{
		print STDERR "WARNING: no file for $chr $infile\n";
	}
}

quantile(\%CoverageControl, \%CoverageMin, $quantile);

foreach my $size (keys(%CoverageMin)){
	print "$size\t$CoverageMin{$size}\t".($#{$CoverageControl{$size}}+1)."\n";
}
# #To remove only for test
# $CoverageMin{'500000'}=0.000259154929577465;
# $CoverageMin{'2000000'}=9.62406015037594e-05;
# $CoverageMin{'0'}=0.000285714285714286;
# $CoverageMin{'1000000'}=0.000168527918781726;
# $CoverageMin{'1500000'}=0.00012265625;
# $CoverageMin{'2500000'}=8.22810590631365e-05;
# 500000  0.000259154929577465    2932758
# 2000000 9.62406015037594e-05    2649634
# 0       0.000285714285714286    1103828
# 1000000 0.000168527918781726    2843878
# 1500000 0.00012265625   2675438
# 2500000 8.22810590631365e-05    1546218



for (my $i_chr=1; $i_chr <= 24; $i_chr ++){
	my $chr = 'chr'.$i_chr;
	@Header = ();
	$chr = 'chrX' if ($i_chr == 23);
	$chr = 'chrY' if ($i_chr == 24);
	my $infile = $infile_name.'_'.$chr.'_enrichment.out';
	if (-e $infile){
		print "read $infile\n";
		# compute ratio
		my %TAD = ();
		@Header = ();
		open IN,$infile;
		while (<IN>){
			chomp($_);
			if ($#Header < 0){
				@Header = split(/\t/,$_);
			}
			else{
				my %Column = ();
				readLine($_, \@Header, \%Column);
				# Chr     Start   End     Coverage_X      Coverage_X_control      Coverage_Y      Coverage_Y_control      nbX_ref nbX_control     nbY_ref nbY_control
				# chr18   210000  465000  151     196     259     195     3.0     3.7     5.1     3.7
				my $size = $Column{'End'} - $Column{'Start'};	
				if ($size >= $delta_diagonale*$resolution) {
					my $size2 = sprintf("%.0f",($size-$delta_diagonale*$resolution)/$round)*$round;
					# print "$size2\n";
					# exit(0);
					#warning : control longer than ref
					$Column{'Coverage_X_control'} = $size*$CoverageMin{$size2} if ($Column{'Coverage_X_control'} <= $size*$CoverageMin{$size2});
					$Column{'Coverage_Y_control'} = $size*$CoverageMin{$size2} if ($Column{'Coverage_Y_control'} <= $size*$CoverageMin{$size2});
					my $ratioX = sprintf("%.1f",$Column{'Coverage_X'}/$Column{'Coverage_X_control'});
					my $ratioY =  sprintf("%.1f",$Column{'Coverage_Y'}/$Column{'Coverage_Y_control'});

					# keep if enrichment in one side
					my @Enriched = ();
					push(@Enriched, 'X') if (($ratioX >= $min_ratio) && ($Column{'nbX_ref'}>=$min_reads_per_bin));
					push(@Enriched, 'Y') if (($ratioY >= $min_ratio) && ($Column{'nbY_ref'}>=$min_reads_per_bin));
					# push(@Enriched, 'X') if (($ratioX >= $min_ratio) && ($Column{'nbX_ref'}>=$CoverageMax{$size}));
					# push(@Enriched, 'Y') if (($ratioY >= $min_ratio) && ($Column{'nbY_ref'}>=$CoverageMax{$size}));				
					if ($#Enriched >= 0){
						if (($#Enriched > 0) || (($#Enriched == 0) && ((($Enriched[0] eq 'X') && ($ratioX >= 1+$min_ratio) && ($Column{'nbX_ref'} >= $min_reads_per_bin)) || (($Enriched[0] eq 'Y') && ($ratioY >= 1+$min_ratio) && ($Column{'nbY_ref'} >= $min_reads_per_bin)) ))){
							$TAD{$Column{'Start'}}{$Column{'End'}+$resolution}{'enriched'} = join(',',sort(@Enriched));
							$TAD{$Column{'Start'}}{$Column{'End'}+$resolution}{'ratioX'} = $ratioX;
							$TAD{$Column{'Start'}}{$Column{'End'}+$resolution}{'ratioY'} = $ratioY;
							foreach my $info (@Info){
								$TAD{$Column{'Start'}}{$Column{'End'}+$resolution}{$info} = $Column{$info};
							}
						}
					}
				}
			}
		}
		close(IN);

		# print selected TADs
		@Start = sort { $a <=> $b} keys(%TAD);
		foreach my $start (@Start){
			@End = sort { $a <=> $b} keys(%{$TAD{$start}});
			foreach my $end (@End){
				print OUT "$chr\t$start\t$end\t$chr:$start-$end\t$TAD{$start}{$end}{'nbX_ref'}\t$TAD{$start}{$end}{'nbY_ref'}\t".$TAD{$start}{$end}{'ratioX'}."\t".$TAD{$start}{$end}{'ratioY'}."\t".$TAD{$start}{$end}{'enriched'}."\n";
			}
		}
	}
}
close(OUT);
print "Write $outfile\n";

##############
# Function
##############
sub readLine{
	my ($line, $refHeader, $refColumn) = @_;
	my @Columns = split(/\t/,$line);
	for (my $i = 0; $i<= $#Columns; $i ++){
		if ($i <= $#$refHeader){
			$$refColumn{$$refHeader[$i]} = $Columns[$i];
		}
		else{
			$$refColumn{$i} = $Columns[$i];
		}
	}
}

sub computeVarianceAndAverage{
	my ($refControl) = @_;
	my $moyenne = 0;
	my $variance = 0;
	my $nb_data = 0;
	# print join(',',@$refControl)."\n";
	foreach my $coverage (@$refControl){
		# die ("Error : $coverage\n") unless ($coverage =~ /^[0-9\-\.e]*$/);
		$moyenne += $coverage;
		$nb_data ++;
	}
	$moyenne = $moyenne/$nb_data;
	foreach my $coverage (@$refControl){
		$variance += ($coverage-$moyenne)*($coverage-$moyenne);
	}

	return($moyenne,sqrt($variance/$nb_data));
}

sub min{
	my ($refTable) = @_;
	my @Sort = sort { $a <=> $b} @$refTable;
	return($Sort[0]);
}

sub quantile{
	my ($refIn, $refOut, $quantile) = @_;
	foreach my $size (keys(%$refIn)){
		my @Value = sort { $a <=> $b} @{$$refIn{$size}};
		my $pos = sprintf("%.0f",($#Value+1)*$quantile);
		$pos = 1 if ($pos < 1);
		$$refOut{$size} = $Value[$pos-1];
		# print "$size\t$$refOut{$size}\n";
	}
}
