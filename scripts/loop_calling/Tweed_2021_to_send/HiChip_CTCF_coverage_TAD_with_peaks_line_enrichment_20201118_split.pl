#!/usr/bin/perl

use strict;
use warnings;
use Getopt::Long;
# use Math::Derivative qw(Derivative1 Derivative2);
# Pb in length normalisation : used length different than ref tad
# compare coverage tot (0) with cov tot (-10) ==> norm. / length X and Y
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
my $delta_bin_peaks = 10; ############### 2 before 
my $chr = '';
my $line = '';
my %Coverage_Y = ();
my %Delta_Coverage = ();
my $coverage1 = 0;
my $start_split = 0;
my $end_split = 0;
##############
# Programs
##############


# filter uniquement sur intrac chr
# suprimer very closed
# transformer count
my $result =  GetOptions ("i=s" => \$infile_pos, "j=s" => \$infile_coverage1,  "p=s" => \$infile_peaks,  "o=s" => \$outfile,  "c=s" => \$chr,"l=s" => \$line,"m=s" => \$min_peak_intensity,"d=s" =>  \$delta_bin_diagonal,"e=s" =>  \$delta_bin_peaks,"a=s" =>  \$start_split,"b=s" =>  \$end_split);

die("No infile_pos $infile_pos\n") unless (-e $infile_pos);
die("No infile_coverage1 $infile_coverage1\n") unless (-e $infile_coverage1);
die("No infile_peaks $infile_peaks\n") unless (-e $infile_peaks);

if (exists($ENV{'TMPDIR'})){
    $tmpdir = $ENV{'TMPDIR'};
}

$tmp_file = $tmpdir.'HiChip_TAD_insulationScore_coverage_20180214.tmp';

# compare bin with CTCF peaks
# faire un fichier tmp pour chaque chr en tenant compte de intensity
extract_bin_with_ctcf($infile_peaks,$infile_pos,$tmp_file, $min_peak_intensity, \%Ctcf_bin,$chr,$start_split,$end_split);
die("No bin selected\n") if (keys(%Ctcf_bin) == 0);

extract_bin_chr($infile_pos, \%Chr_bin,$chr);
die("No good chr selected $chr\n") if (keys(%Chr_bin) == 0);

my $resolution = extract_resolution($infile_pos);
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
open OUT,'>'.$outfile.'_'.$chr.'_enrichment.out' || die("Error in creation of outfile\n");
print OUT "Chr\tStart\tEnd\tCoverage_X\tCoverage_X_control\tCoverage_Y\tCoverage_Y_control\tnbX_ref\tnbX_control\tnbY_ref\tnbY_control\n";
readCoverage($chr,$tmp_file1,\%Ctcf_bin,$resolution,$coverage1,$Chr_bin{$chr}{'min'},$Chr_bin{$chr}{'max'},$delta_pos_peaks,$delta_bin_diagonal, $delta_bin_peaks);	
close(OUT);
print "Write ".$outfile.'_'.$chr.'_enrichment.out'."\n";


##############
# Function
##############

sub extract_bin_with_ctcf{
	my ($infile_peaks,$infile_pos,$tmp_file,$min_peak_intensity,$refPeaks,$chr_ref,$start_split,$end_split) = @_;
	my $tmp_file_peaks = $tmp_file.'_peaks';
	my $resolution = 0;
	# ctcf : tmp file
	system('awk -F"\t" \'($1=="'.$chr_ref.'") && ($7>='.$min_peak_intensity.') && (($2 >= '.$start_split.') && ($2 <= '.$end_split.')) || (($3 >= '.$start_split.') && ($3 <= '.$end_split.')) {print $0}\' '.$infile_peaks.' > '.$tmp_file_peaks);
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
						}
					}
				# }
				$i_bin2 ++;
			}
			if (keys(%End_Tad) > 0){			
				my @End_Tad = sort { $a <=> $b} keys(%End_Tad);
				my $last_end = $End_Tad[$#End_Tad];

				my %Coverage = ();
				die ("error with $last_end\n$i_bin1\n".keys(%End_Tad)."\t$#End_Tad\n") unless(defined($last_end));
				readCoverageTad ($infile_coverage1,$start,$last_end,$delta_bin_peaks,$delta_bin_diagonal,\%End_Tad,\%Coverage);
				
				for (my $i_end = 0; $i_end <= $#End_Tad; $i_end ++){
					my $end = $End_Tad[$i_end];
					my $i_x_control =  mediane_control('X', \%Coverage, $start, $end, $delta_bin_peaks, $delta_bin_diagonal);
					my $i_y_control =  mediane_control('Y', \%Coverage, $start, $end, $delta_bin_peaks, $delta_bin_diagonal);
					my $nbX_ref = sprintf("%.1f",$Coverage{$end}{'X'}{$start}/($end-$start-$delta_bin_diagonal));
					my $nbX_control = sprintf("%.1f",$Coverage{$end}{'X'}{$start-$i_x_control}/($end-$start-$delta_bin_diagonal));
					my $nbY_ref = sprintf("%.1f",$Coverage{$end}{'Y'}{$end}/($end-$start-$delta_bin_diagonal));
					my $nbY_control = sprintf("%.1f",$Coverage{$end}{'Y'}{$end+$i_y_control}/($end-$start-$delta_bin_diagonal));
					print OUT "$chr\t".(($start-$bin_min_chr)*$resolution_coverage)."\t".(($end -$bin_min_chr)*$resolution_coverage)."\t$Coverage{$end}{'X'}{$start}\t".$Coverage{$end}{'X'}{$start-$i_x_control}."\t$Coverage{$end}{'Y'}{$end}\t".$Coverage{$end}{'Y'}{$end+$i_y_control}."\t$nbX_ref\t$nbX_control\t$nbY_ref\t$nbY_control\n";
					
				}
			}
		# }
	}
}

sub mediane_control{
	my ($line, $refIn, $start, $end, $delta, $delta_bin_diagonal) = @_;
	my @Value = ();
	my %Res = ();
	my $control = 0;
	for (my $i = 1 ; $i <= $delta; $i ++){
	
		if ($line eq 'X'){
			die("No value found for X $start $i :".($start-$i)."\n".join(',',keys(%{$$refIn{$end}{'X'}}))."\n") unless (exists($$refIn{$end}{'X'}{$start-$i}));
			$control = $$refIn{$end}{'X'}{$start-$i}/($end-$start-$delta_bin_diagonal);
		}
		else{
			$control = $$refIn{$end}{'Y'}{$end+$i}/($end-$start-$delta_bin_diagonal);
		}
		$Res{$control} = $i;
		push (@Value, $control);
	}
	@Value = sort { $a <=> $b} @Value;
	my $median = sprintf("%.0f",($#Value+1)/2);
	my $i_median = $Res{$Value[$median]};
	return $i_median;
}

# Control = same length than ref
sub readCoverageTad{
	my ($infile_coverage,$start,$last_end,$delta_bin,$delta_bin_diagonal,$ref_Tad, $ref_Coverage) = @_;
	foreach my $end (keys(%$ref_Tad)){
		$$ref_Coverage{$end}{'X'}{$start} = 0;
		for (my $i = 0 ; $i <= $delta_bin; $i ++){
			$$ref_Coverage{$end}{'X'}{$start-$i} = 0;
			$$ref_Coverage{$end}{'Y'}{$end+$i} = 0;	
		}
	}
	open IN,$infile_coverage;
	while (<IN>){
		chomp($_);
		my ($bin1, $bin2, $coverage ) = split(/\t/,$_);
		if ($bin2-$bin1>=$delta_bin_diagonal){ 
			# if (($bin1 >= $start-$delta_bin) && ($bin2 <= $end+$delta_bin)){
				foreach my $end (keys(%$ref_Tad)){
					if ((($bin1 == $start) || ($bin1 >= $start-$delta_bin)) && ($bin2 <= $end)){
						my $i = $start-$bin1; # delta
						if ($bin2 <= $end-$i){
							$$ref_Coverage{$end}{'X'}{$bin1} += $coverage ;	
						}
					}
					if ((($bin2 == $end) || ($bin2 <= $end+$delta_bin)) && ($bin1 >= $start)){
						my $i = $bin2-$end; # delta
						if ($bin1 >= $start+$i){
							$$ref_Coverage{$end}{'Y'}{$bin2} += $coverage;	
						}
					}
				}
			# }
		}
		last() if ($bin1 > $last_end+$delta_bin);
	}
	close(IN);

}
