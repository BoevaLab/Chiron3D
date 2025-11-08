#!/usr/bin/perl

use strict;
use warnings;
use Getopt::Long;
use Math::Derivative qw(Derivative1 Derivative2);

##############
# Declaration
##############

my $infile_pos = '';
my $infile_coverage1 = '';
my $infile_peaks = '';
my $tmpdir = './';
my $tmp_file = '';
my $outfile = '';
my %Ctcf_bin = ();
my %Chr_bin = ();
my $min_peak_intensity = 10;
my $delta_pos_peaks = 2500000 ; # 3 Mb
my $delta_bin_diagonal = 15;
my $delta_bin_peaks = 10;
my $chr = '';
my $line = '';
my %Coverage_Y = ();
my %Delta_Coverage = ();
my $coverage1 = 0;
my $resolution = 'na';
##############
# Programs
##############

# filter uniquement sur intrac chr
# suprimer very closed
# transformer count
my $result =  GetOptions ("i=s" => \$infile_pos, "j=s" => \$infile_coverage1,  "p=s" => \$infile_peaks,  "o=s" => \$outfile,  "c=s" => \$chr,"l=s" => \$line,"m=s" => \$min_peak_intensity,"d=s" =>  \$delta_bin_diagonal,"e=s" =>  \$delta_bin_peaks,"t=s" =>  \$tmpdir,"r=s" =>  \$resolution);

die("No infile_pos $infile_pos\n") unless (-e $infile_pos);
die("No infile_coverage1 $infile_coverage1\n") unless (-e $infile_coverage1);
die("No infile_peaks $infile_peaks\n") unless (-e $infile_peaks);

if (exists($ENV{'TMPDIR'})){
    $tmpdir = $ENV{'TMPDIR'};
}

$tmp_file = $tmpdir.'HiChip_TAD_insulationScore_coverage_20180214.tmp';

# compare bin with CTCF peaks
# faire un fichier tmp pour chaque chr en tenant compte de intensity
extract_bin_with_ctcf($infile_peaks,$infile_pos,$tmp_file, $min_peak_intensity, \%Ctcf_bin,$chr);
die("No bin selected\n") if (keys(%Ctcf_bin) == 0);

extract_bin_chr($infile_pos, \%Chr_bin,$chr);
die("No good chr selected $chr\n") if (keys(%Chr_bin) == 0);

if ($resolution eq 'na'){
	$resolution = extract_resolution($infile_pos);
}
print STDOUT "Coverage resolution = $resolution\n";


my $tmp_file1 = $tmp_file.'1';
print STDOUT "Look for bins with CTCF...\n";

if (-e $tmp_file1){
	open IN, $outfile.'_'.$chr.'_coverage_tot.txt';
	$_ = <IN>;
	chomp($_);
	my @Tmp = split(/\t/,$_);
	$coverage1 = $Tmp[1];
	close(IN);
}
else{
	$coverage1 = readTotalCoverage($infile_coverage1, $Chr_bin{$chr}{'min'},$Chr_bin{$chr}{'max'}, $tmp_file1);
	unless (-e $outfile.'_'.$chr.'_coverage_tot.txt'){
		open OUT,'>'.$outfile.'_'.$chr.'_coverage_tot.txt';
		print OUT "Sample1\t$coverage1\n";
		close(OUT);
	}
}

print STDOUT "Compute TAD...\n";
open OUT,'>'.$outfile.'_'.$chr.'_derivative.out' || die("Error in creation of outfile\n");
print OUT "Chr\tStart\tEnd\tLine\tReads_per_bin";
for (my $i_delta = - $delta_bin_peaks; $i_delta <= $delta_bin_peaks; $i_delta ++){
	if ($i_delta < 0){
		print OUT "\tDeriv_m".abs($i_delta);
	}
	else{
		print OUT "\tDeriv_p$i_delta";
	}
}
for (my $i_delta = - $delta_bin_peaks; $i_delta <= $delta_bin_peaks; $i_delta ++){
	if ($i_delta < 0){
		print OUT "\tCov_m".abs($i_delta);
	}
	else{
		print OUT "\tCov_p$i_delta";
	}
}
print OUT "\n";
readCoverage($chr,$tmp_file1,\%Ctcf_bin,$resolution,$coverage1,$Chr_bin{$chr}{'min'},$Chr_bin{$chr}{'max'},$delta_pos_peaks,$delta_bin_diagonal, $delta_bin_peaks);	
close(OUT);
print "Write ".$outfile.'_'.$chr.'_derivative.out'."\n";


##############
# Function
##############

sub extract_bin_with_ctcf{
	my ($infile_peaks,$infile_pos,$tmp_file,$min_peak_intensity,$refPeaks,$chr_ref) = @_;
	my $tmp_file_peaks = $tmp_file.'_peaks';
	my $resolution = 0;
	# ctcf : tmp file
	system('awk -F"\t" \'($1=="'.$chr_ref.'") && ($7>='.$min_peak_intensity.') {print $0}\' '.$infile_peaks.' > '.$tmp_file_peaks);
	# compare with peaks	
	print "bedtools intersect -wo -a $infile_pos -b $tmp_file_peaks > $tmp_file\n";
	system("bedtools intersect -wo -a $infile_pos -b $tmp_file_peaks > $tmp_file");
	open IN,$tmp_file;
	while (<IN>){
		chomp($_);
		my ($chr1,$start1,$end1,$bin1,@Peaks) = split(/\t/,$_);
		my $intensity = $Peaks[6];
		# version before 15/05/20
		# my $strand = $Peaks[10];
		# $$refPeaks{$chr1}{$bin1}{$strand} = '';
		$$refPeaks{$chr1}{$bin1} = '';
	}
	close(IN);

}

sub extract_bin_chr{
	my ($infile_pos, $ref_chr,$chr_ref) = @_;
	print "extract_bin_chr...\n";
	open IN,$infile_pos;
	while (<IN>){
		chomp($_);
		my ($chr1,$start1,$end1,$bin1) = split(/\t/,$_);
		if ($chr_ref eq $chr1){
			if (exists($$ref_chr{$chr1})){
				$$ref_chr{$chr1}{'max'} = $bin1;
			}
			else{
				$$ref_chr{$chr1}{'min'} = $bin1;
			}
		}
		else{
			last() if (exists($$ref_chr{$chr_ref}));
		}
	}
	close(IN);
}

sub extract_resolution{
	my ($infile_pos) = @_;
	print "Resolution...\n";
	open IN,$infile_pos;
	$_ = <IN>;
	my ($chr1,$start1,$end1,$bin1) = split(/\t/,$_);
	$resolution = $end1 - $start1;
	close(IN);
	return($resolution);
}

sub readTotalCoverage{
	my ($infile_coverage, $min, $max, $outfile) = @_;
	my $total = 0;
	my %Bin = ();
	print "Write tmpfile $outfile\n";
	open TMP,">".$outfile;
	open IN,$infile_coverage;
	while (<IN>){
		chomp($_);
		my ($bin1, $bin2, $coverage ) = split(/\t/,$_);
		if (($bin1 >= $min) && ($bin1 <= $max) && ($bin2 >= $min) && ($bin2 <= $max)){
			$total += $coverage;
			print TMP "$_\n";
		}
		last() if ($bin1 > $max);
	}
	close(IN);
	close(TMP);
	return($total);
}

	
sub readCoverage{
	my ($chr, $infile_coverage1,$refCtcf_bin,$resolution_coverage,$coverage_chr1,$bin_min_chr,$bin_max_chr,$delta_pos_peaks,$delta_bin_diagonal,$delta_bin_peaks) = @_;

	my @Bin = sort { $a <=> $b} keys(%{$$refCtcf_bin{$chr}});
	print $#Bin." bins with CTCF ($chr)\n";
	# print join(",",@Bin)."\n";
	my $delta_bin = int($delta_pos_peaks/$resolution_coverage); # 3 Mb
	my %TadY = ();
	my %CoverageY = ();
	# foreach bin with CTCF peaks (filter peaks intensity ??)
	# enlever 20 kb autour de diagonale
	for (my $i_bin1 = 0; $i_bin1 < $#Bin; $i_bin1 ++){
		# motif +
		# if (exists($$refCtcf_bin{$chr}{$Bin[$i_bin1]}{'+'})){
			my $start = $Bin[$i_bin1];		
			# if ((($start-$bin_min_chr)*5000 == 44235000) || (($start-$bin_min_chr)*5000 == 55225000)){
			my $end_min = $Bin[$i_bin1];
			my $end_max = $Bin[$i_bin1]+$delta_bin+$delta_bin_peaks;
			$end_min = $bin_min_chr if ($end_min < $bin_min_chr);
			$end_max = $bin_max_chr if ($end_max > $bin_max_chr);
			my %End_Tad = ();
			my $i_bin2 = 0;
			while (($i_bin2 <= $#Bin) && ($Bin[$i_bin2] <=  $end_max)){
				# if (exists($$refCtcf_bin{$chr}{$Bin[$i_bin2]}{'-'})){
					if (($Bin[$i_bin2] >  $end_min) && ($Bin[$i_bin2] <=  $end_max)){
						if (($Bin[$i_bin2]-$start)>$delta_bin_diagonal){
							$End_Tad{$Bin[$i_bin2]} = '';
							$TadY{$Bin[$i_bin2]}{$start} = '';
						}
					}
				# }
				$i_bin2 ++;
			}
			if (keys(%End_Tad) > 0){			
				my @End_Tad = sort { $a <=> $b} keys(%End_Tad);
				my $last_end = $End_Tad[$#End_Tad];

				my %Coverage1 = ();
				my %DeltaX = ();
				die ("error with $last_end\n$i_bin1\n".keys(%End_Tad)."\t$#End_Tad\n") unless(defined($last_end));
				readCoverageTadX ($infile_coverage1,$start,$last_end,\%DeltaX);
	
				my @Pos = ();
				my @X = ();
				my $coverage = 0;
				for (my $i1 = $start+$delta_bin_diagonal; $i1 <= $last_end; $i1 ++){
					my $pos = ($i1-$start);
					if (exists($DeltaX{$start}{$i1})){
						$coverage += $DeltaX{$start}{$i1};
					}
					# else 0
					push(@Pos,$pos);
					push(@X,$coverage);		
				}
				my @DerivativeX = Derivative2(\@Pos,\@X) ;
				for (my $i_end = 0; $i_end <= $#Pos; $i_end ++){				
					if (exists($End_Tad{($Pos[$i_end]+$start)})){					
						print OUT "$chr\t".(($start-$bin_min_chr)*$resolution_coverage)."\t".(($Pos[$i_end]+$start-$bin_min_chr)*$resolution_coverage)."\tX\t".sprintf("%.2f",$X[$i_end]/($Pos[$i_end]-$delta_bin_diagonal));
						for (my $i_delta = $i_end - $delta_bin_peaks; $i_delta <= $i_end + $delta_bin_peaks; $i_delta ++){
							if (($i_delta >= 0) && ($i_delta <= $#Pos)){
								$DerivativeX[$i_delta] = sprintf("%.2f",$DerivativeX[$i_delta]) if (abs($DerivativeX[$i_delta]) >= 0.1);
								print OUT "\t$DerivativeX[$i_delta]";
							}
							else{
								print OUT "\tNA";
							}						
						}
						for (my $i_delta = $i_end - $delta_bin_peaks; $i_delta <= $i_end + $delta_bin_peaks; $i_delta ++){
							if (($i_delta >= 0) && ($i_delta <= $#Pos)){
								print OUT "\t$X[$i_delta]";
							}
							else{
								print OUT "\tNA";
							}						
						}
						print OUT "\n";							
					}
				}
			}
		# }
	}

	readCoverageTadY($infile_coverage1,$delta_bin,\%TadY,\%CoverageY);
	my @End = sort { $a <=> $b} keys(%TadY);	
	for (my $i_bin1 = 0; $i_bin1 < $#End; $i_bin1 ++){
		my $end_tad = $End[$i_bin1];
		my @Start_Tad = sort { $a <=> $b} keys(%{$TadY{$end_tad}});
		my @Pos = ();
		my @Y = ();
		my $coverage = 0;
			
		for (my $i_start = $end_tad-$delta_bin_diagonal; $i_start >= $Start_Tad[0]; $i_start = $i_start -1){
			# my $pos = ($i_start-$Start_Tad[0]);
			my $pos = ($end_tad-$i_start);
			if (exists($CoverageY{$end_tad}{$i_start})){
				$coverage += $CoverageY{$end_tad}{$i_start};
			}
			# else 0
			push(@Pos,$pos);
			push(@Y,$coverage);		
		}
		my @DerivativeY = Derivative2(\@Pos,\@Y);
		for (my $i_start = 0; $i_start <= $#Pos; $i_start ++){
			my $start_tmp = $end_tad-$Pos[$i_start];
			if (exists($TadY{$end_tad}{$start_tmp})){					
				print OUT "$chr\t".(($start_tmp-$bin_min_chr)*$resolution_coverage)."\t".(($end_tad-$bin_min_chr)*$resolution_coverage)."\tY\t".sprintf("%.2f",$Y[$i_start]/($Pos[$i_start]-$delta_bin_diagonal));
				for (my $i_delta = $i_start - $delta_bin_peaks; $i_delta <= $i_start + $delta_bin_peaks; $i_delta ++){
					if (($i_delta >= 0) && ($i_delta <= $#Pos)){
						$DerivativeY[$i_delta] = sprintf("%.2f",$DerivativeY[$i_delta]);
						print OUT "\t$DerivativeY[$i_delta]";
					}
					else{
						print OUT "\tNA";
					}
				}
				for (my $i_delta = $i_start - $delta_bin_peaks; $i_delta <= $i_start + $delta_bin_peaks; $i_delta ++){
					if (($i_delta >= 0) && ($i_delta <= $#Pos)){
						print OUT "\t$Y[$i_delta]";
					}
					else{
						print OUT "\tNA";
					}
				}
				print OUT "\n";
			}
		}
	}
}

sub readCoverageTadX{
	my ($infile_coverage,$start,$end,$ref_delta) = @_;

	open IN,$infile_coverage;
	while (<IN>){
		chomp($_);
		my ($bin1, $bin2, $coverage ) = split(/\t/,$_);
		if (($bin1 == $start) && ($bin2 <= $end)){
			$$ref_delta{$start}{$bin2} = $coverage;	
		}
		last() if ($bin1 > $start);
	}
	close(IN);

}

sub readCoverageTadY{
	my ($infile_coverage,$delta_max,$refTad,$ref_delta) = @_;
	open IN,$infile_coverage;
	while (<IN>){
		chomp($_);
		my ($bin1, $bin2, $coverage ) = split(/\t/,$_);
		if ($bin2-$bin1 <= $delta_max){
			if (exists($$refTad{$bin2})){
				$$ref_delta{$bin2}{$bin1} = $coverage;
			}
		}
	}
	close(IN);

}