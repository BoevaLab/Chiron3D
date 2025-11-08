#!/usr/bin/perl

use strict;
use warnings;
use Getopt::Long;

# keep if enrichment in one side

##############
# Declaration
##############

my $tmpdir = './';
my $infile_deriv = '';
my $infile_enrich = '';
my $outfile = '';
my %Tad = ();
my %Remove = ();
my %Color = ();
$Color{3} = '153,204,255'; #light blue
$Color{2} = '0,0,255'; # blue
$Color{1} = '0,0,0'; # black
my $overlap = 0.95;
my %OneSide = ();
my %BothSide = ();
my $delta_both = 20;
my $resolution = 5000;
##############
# Programs
##############

my $result =  GetOptions ("i=s" => \$infile_deriv, "j=s" => \$infile_enrich, "o=s" => \$outfile, "r=s" => \$resolution);

die("No infile_deriv $infile_deriv\n") unless (-e $infile_deriv);
die("No infile_enrich $infile_enrich\n") unless (-e $infile_enrich);

$delta_both = $delta_both * $resolution;
if (exists($ENV{'TMPDIR'})){
    $tmpdir = $ENV{'TMPDIR'};
}

my $tmp_file = $tmpdir.'HiChip_CTCF_coverage_TAD_merge_prediction_20180221.tmp';
my $tmp_file1 = $tmp_file.'1';
my $tmp_file2 = $tmp_file.'2';

#########


print "Extract exactly same TADs\n";
# chr18   265000  625000  chr18:265000-625000     2.6     5.8     35      10      13      79      10      39      chr18   265000  625000  chr18:265000-625000     2.6     5.8     1.5     7.2     360000
print "bedtools intersect -wo -r -f 1.0 -a $infile_deriv -b $infile_enrich > $tmp_file1\n";
system("bedtools intersect -wo -r -f 1.0 -a $infile_deriv -b $infile_enrich > $tmp_file1");
open  IN,$tmp_file1;
while (<IN>){
	chomp($_);
	my ($chr1,$start1, $end1, $name1,$nbX1, $nbY1,$deltaXminus,$covX,$deltaXplus,$deltaYminus,$covY,$deltaYplus,$enrich_status1,$chr2,$start2, $end2, $name2,$nbX2, $nbY2,$enrichX,$enrichY,$enrich_status2,$overlap)= split(/\t/,$_);
	my $enrich_status = status($enrich_status1,$enrich_status2);
	my @Info = join("\t",($chr1,$start1, $end1, $name1,1,$nbX1, $nbY1,$deltaXminus,$covX,$deltaXplus,$deltaYminus,$covY,$deltaYplus,$enrichX,$enrichY,$enrich_status)) ;
	if ($enrich_status ne 'NA'){
		# print "1 $enrich_status\t$_\n" if ($_ =~ /chr12:89935000-108090000/);
		if ($enrich_status ne 'X,Y'){
			# if extrusion, take the highr
			if ($enrich_status eq 'X'){
				if (exists($OneSide{$chr1}{$enrich_status}{'start'}{$start1})){
					print "1 a\t$enrichX > $OneSide{$chr1}{$enrich_status}{$start1}{'enrich'}\n" if ($_ =~ /chr12:104775000-105450000/);
					if ($OneSide{$chr1}{$enrich_status}{'start'}{$start1}{'end'} < $end1){
						# print "1 a a\n" if ($_ =~ /chr12:89935000-108090000/);
						$OneSide{$chr1}{$enrich_status}{'start'}{$start1}{'enrich'} = $enrichX ;
						$OneSide{$chr1}{$enrich_status}{'start'}{$start1}{'end'} = $end1 ;
						$OneSide{$chr1}{$enrich_status}{'start'}{$start1}{'info'} = join("\t",@Info) ;
					}
				}
				else{
					# print "1 b\n" if ($_ =~ /chr12:89935000-108090000/);
					$OneSide{$chr1}{$enrich_status}{'start'}{$start1}{'enrich'} =  $enrichX;
					$OneSide{$chr1}{$enrich_status}{'start'}{$start1}{'end'} = $end1 ;
					$OneSide{$chr1}{$enrich_status}{'start'}{$start1}{'info'} = join("\t",@Info) ;
				}
			}
			else{
				if (exists($OneSide{$chr1}{$enrich_status}{$end1})){
					if ($OneSide{$chr1}{$enrich_status}{'end'}{$end1}{'start'} > $start1){
						$OneSide{$chr1}{$enrich_status}{'end'}{$end1}{'enrich'} = $enrichY ;
						$OneSide{$chr1}{$enrich_status}{'end'}{$end1}{'start'} = $start1 ;
						$OneSide{$chr1}{$enrich_status}{'end'}{$end1}{'info'} = join("\t",@Info) ;
					}
				}
				else{
					$OneSide{$chr1}{$enrich_status}{'end'}{$end1}{'enrich'} = $enrichY;
					$OneSide{$chr1}{$enrich_status}{'end'}{$end1}{'start'} = $start1 ;
					$OneSide{$chr1}{$enrich_status}{'end'}{$end1}{'info'} = join("\t",@Info) ;
				}
			}
		}
		else{
			$Tad{$name1} = join("\t",@Info) ;
			if (exists($BothSide{$chr1}{'end'}{$end1})){
				if ($start1 < $BothSide{$chr1}{'end'}{$end1}{'start'}){
					$BothSide{$chr1}{'end'}{$end1}{'enrich'} = $enrichY;
					$BothSide{$chr1}{'end'}{$end1}{'start'} = $start1;
				}
			}
			else{
				$BothSide{$chr1}{'end'}{$end1}{'enrich'} = $enrichY;
				$BothSide{$chr1}{'end'}{$end1}{'start'} = $start1;
			}
			if (exists($BothSide{$chr1}{'start'}{$start1}{'end'})){
				if ($end1 > $BothSide{$chr1}{'start'}{$start1}{'end'}){
					$BothSide{$chr1}{'start'}{$start1}{'enrich'} = $enrichX;
					$BothSide{$chr1}{'start'}{$start1}{'end'} = $end1;
				}
			}
			else{
				$BothSide{$chr1}{'start'}{$start1}{'enrich'} = $enrichX;
				$BothSide{$chr1}{'start'}{$start1}{'end'} = $end1;
			}
		}
	}
}
close(IN);

# print "test1 ".$BothSide{'chr12'}{'start'}{89935000}{'end'}."\n";
# print "test2 ".$BothSide{'chr12'}{'end'}{91350000}{'start'}."\n";

# if extrusion in one line only, keep tad if no overlap with tad X,Y (same start or same end)
foreach my $chr (keys(%OneSide)){
	foreach my $status (keys(%{$OneSide{$chr}})){
		foreach my $boundary (keys(%{$OneSide{$chr}{$status}})){
			foreach my $pos (keys(%{$OneSide{$chr}{$status}{$boundary}})){
				# print "\n2 $boundary\n" if ($pos == 89935000);
				if (exists($OneSide{$chr}{$status}{$boundary}{$pos}{'start'})){
					my $start = $OneSide{$chr}{$status}{$boundary}{$pos}{'start'};
					my $end = $pos;
					# print "2a $start $end\n" if ($start == 89935000);
					my @Line = split(/\t/,$OneSide{$chr}{$status}{$boundary}{$pos}{'info'});
					if (exists($BothSide{$chr}{'end'}{$end}{'start'})){
						# print "2a a\n" if ($start == 89935000);
						# higher and more enriched
						if ($BothSide{$chr}{'end'}{$end}{'start'} > $start+$delta_both){
						# if (($BothSide{$chr}{'end'}{$end}{'start'} > $start+$delta_both) && ($BothSide{$chr}{'end'}{$end}{'enrich'} <= $OneSide{$chr}{$status}{$boundary}{$pos}{'enrich'})){
							$Tad{$Line[3]} = $OneSide{$chr}{$status}{$boundary}{$pos}{'info'} ;
						}
					}
					else{
						# print "2a b\n" if ($start == 89935000);
						$Tad{$Line[3]} = $OneSide{$chr}{$status}{$boundary}{$pos}{'info'} ;					
					}
				}
				if (exists($OneSide{$chr}{$status}{$boundary}{$pos}{'end'})){
					my $end = $OneSide{$chr}{$status}{$boundary}{$pos}{'end'};
					my $start = $pos;
					# print "2b $start $end\n" if ($start == 89935000);
					my @Line = split(/\t/,$OneSide{$chr}{$status}{$boundary}{$pos}{'info'});
					if (exists($BothSide{$chr}{'start'}{$start}{'end'})){
						# higher and more enriched
						# print "na\n" if ($start == 89935000);
						if ($BothSide{$chr}{'start'}{$start}{'end'} < $end-$delta_both) {
						# if (($BothSide{$chr}{'start'}{$start}{'end'} < $end-$delta_both) && ($BothSide{$chr}{'start'}{$start}{'enrich'} <= $OneSide{$chr}{$status}{$boundary}{$start}{'enrich'})){
							# print "ok2\n" if ($start == 89935000);
							$Tad{$Line[3]} = $OneSide{$chr}{$status}{$boundary}{$pos}{'info'} ;
						}
					}
					else{
						# print "ok3\n" if ($start == 89935000);
						$Tad{$Line[3]} = $OneSide{$chr}{$status}{$boundary}{$pos}{'info'} ;					
					}
				}
			}
		}
	}
}

print keys(%Tad)." exact tads\n";

open OUT,">".$tmp_file1;
foreach my $tad (keys(%Tad)){
	print OUT "$Tad{$tad}\n";
}

# print "Extract specific TADs\n";
# open IN,$infile_deriv;
# while (<IN>){
	# chomp($_);
	# my ($chr1,$start1, $end1, $name1,$nbX1, $nbY1,$deltaXminus,$covX,$deltaXplus,$deltaYminus,$covY,$deltaYplus,$enrich_status)= split(/\t/,$_);
	# unless (exists($Remove{$name1}) || exists($Tad{$name1})){
		# print OUT join("\t",($chr1,$start1, $end1, $name1,3,$nbX1, $nbY1,$deltaXminus,$covX,$deltaXplus,$deltaYminus,$covY,$deltaYplus,'NA','NA',$enrich_status))."\n";
	# }
# }
# close(IN);

# open IN,$infile_enrich;
# while (<IN>){
	# chomp($_);
	# my ($chr2,$start2, $end2, $name2,$nbX2, $nbY2,$enrichX,$enrichY,$enrich_status)= split(/\t/,$_);
	# unless (exists($Remove{$name2}) || exists($Tad{$name2})){
		# print OUT join("\t",($chr2,$start2, $end2, $name2,3,$nbX2, $nbY2,'NA','NA','NA','NA','NA','NA','NA','NA',$enrich_status))."\n";
	# }
# }
# close(IN);
close(OUT);


print "bedtools sort -i $tmp_file1 > $tmp_file2\n";
system("bedtools sort -i $tmp_file1 > $tmp_file2");



print "Extract very closed TADs\n";
print "bedtools intersect -wo -r -f $overlap -a $tmp_file2 -b $tmp_file2 > $tmp_file1\n";
system("bedtools intersect -wo -r -f $overlap -a $tmp_file2 -b $tmp_file2 > $tmp_file1");

open  IN,$tmp_file1;
while (<IN>){
	chomp($_);
	my ($chr1,$start1, $end1, $name1,$score1,$nbX1, $nbY1,$deltaXminus1,$covX1,$deltaXplus1,$deltaYminus1,$covY1,$deltaYplus1,$enrichX1,$enrichY1,$enrich_status1,$chr2,$start2, $end2, $name2,$score2,$nbX2, $nbY2,$deltaXminus2,$covX2,$deltaXplus2,$deltaYminus2,$covY2,$deltaYplus2,$enrichX2,$enrichY2,$enrich_status2)= split(/\t/,$_);
	my $enrich_status = status($enrich_status1,$enrich_status2);
	if ($name1 ne $name2){
		# 1 bin of difference
		if ((abs($start1-$start2) <= $resolution) && (abs($end1-$end2) <= $resolution)){
			# print "2\t$_\n" if ($_ =~ /chr12:104775000-105450000/);
			if ($enrich_status eq 'X,Y'){
				if (($enrich_status1 eq 'X,Y') && ($enrich_status2 eq 'X,Y')){
					if ($nbX1+$nbY1 > $nbX2+$nbY2){;
						$Remove{$name2} = '';
					}
					else{
						$Remove{$name1} = '';
					}
				}
				else{
					if ($enrich_status1 eq 'X,Y'){
						$Remove{$name2} = '';
					}
					else{
						$Remove{$name1} = '';
					}
				}
			}
			else{
				# keep the best ==> nb reads / bin higher
				if (($nbX1 + $nbY1) >= ($nbX2 + $nbY2)){
					$Remove{$name2} = '' ;
				}
				else{
					$Remove{$name1} = '' ;
				}
			}
		}
	}
}
close(IN);
# exit(0);

print keys(%Remove)." removed tad due to closed\n";

print "Write ".$outfile.".bed\n";
open OUT,">".$outfile.".bed";
open  IN,$tmp_file2;
while (<IN>){
	chomp($_);
	my ($chr1,$start1, $end1, $name1,$score1,$nbX1, $nbY1,$deltaXminus,$covX,$deltaXplus,$deltaYminus,$covY,$deltaYplus,$enrichX,$enrichY,$enrich_status)= split(/\t/,$_);
	unless (exists($Remove{$name1})){
		print OUT "$_\n";
	}
}
close(IN);
close(OUT);

print "Write ".$outfile.".juicebox\n";
open JUICEBOX,">".$outfile.'.juicebox';
print JUICEBOX "chr1\tx1\tx2\tchr2\ty1\ty2\tcolor\tscore\n";
open IN,$outfile.".bed";
while (<IN>){
	chomp($_);
	my ($chr1,$start1, $end1, $name1,$score1,$nbX1, $nbY1,$deltaXminus,$covX,$deltaXplus,$deltaYminus,$covY,$deltaYplus,$enrichX,$enrichY,$enrich_status) = split(/\t/,$_);
	print JUICEBOX "$chr1\t$start1\t$end1\t$chr1\t$start1\t$end1\t$Color{$score1}\tnbX=$nbX1; nbY=$nbY1; deltaXminus=$deltaXminus; covX=$covX; deltaXplus=$deltaXplus; deltaYminus=$deltaYminus; covY=$covY; deltaYplus=$deltaYplus; enrichX=$enrichX; enrichY=$enrichY; enrich_status=$enrich_status\n"
}
close(IN);
close(JUICEBOX);
print "Done\n";






##############
# Function
##############
sub status{
	my ($enrich_status1,$enrich_status2) = @_;
	my $enrich_status = 'NA';
	if ($enrich_status1 eq $enrich_status2){
		$enrich_status = $enrich_status1;
	}
	else{
		if (($enrich_status1 =~ /X/) && ($enrich_status2 =~ /X/)){
			$enrich_status = 'X';
		}
		elsif(($enrich_status1 =~ /Y/) && ($enrich_status2 =~ /Y/)){
			$enrich_status = 'Y';
		}
	}	
	return $enrich_status;
}

sub merge_tads{
	my ( $infile_ref,$overlap,$outfile,$tmp_file,$bedtools_path) = @_;
	my $tmp_file1 = $tmp_file.'1';
	my $tmp_file2 = $tmp_file.'2';
	my $nb_col = 0;
	my %Remove = ();
	my @Line= ();
	
	open IN,$infile_ref;
	$_ = <IN>;
	@Line= split(/\t/,$_);
	$nb_col = $#Line +1;
	close(IN);
	
	print "Sort files\n";
	print "bedtools sort -i $infile_ref > $tmp_file1\n";
	system("bedtools sort -i $infile_ref > $tmp_file1");

	print "Extract very closed TADs\n";
	print "bedtools intersect -wo -r -f $overlap -a $tmp_file1 -b $tmp_file1 > $tmp_file2\n";
	system("bedtools intersect -wo -r -f $overlap -a $tmp_file1 -b $tmp_file1 > $tmp_file2");
	open  IN,$tmp_file2;
	while (<IN>){
		chomp($_);
		@Line = split(/\t/,$_);
		my ($chr1, $start1, $end1, $name1, $nbX1, $nbY1) = ($Line[0],$Line[1],$Line[2],$Line[3],$Line[4],$Line[5],$Line[6]);
		my ($chr2, $start2, $end2, $name2, $nbX2, $nbY2) = ($Line[0+$nb_col],$Line[1+$nb_col],$Line[2+$nb_col],$Line[3+$nb_col],$Line[4+$nb_col],$Line[5+$nb_col],$Line[6+$nb_col]);
		if ($name1 ne $name2){
			# keep the best ==> nb reads / bin higher
			if (($nbX1 + $nbY1) >= ($nbX2 + $nbY2)){
				$Remove{$name2} = '' ;
			}
			else{
				$Remove{$name1} = '' ;
			}
			# print "s1 $name1 / $name2: $_\n" if (( $name1 eq 'chr12:104750000-105500000') || ($name2 eq 'chr12:104750000-105500000'));
		}
	}
	close(IN);
	
	open  OUT,'>'.$tmp_file1;
	open  IN,$infile_ref;
	while (<IN>){
		chomp($_);
		@Line = split(/\t/,$_);
		my ($chr1,$start1, $end1, $name1,$nbX1, $nbY1) = ($Line[0],$Line[1],$Line[2],$Line[3],$Line[4],$Line[5],$Line[6]);
		unless (exists($Remove{$name1})){
			print OUT "$_\n";
			# print "s2 : $_\n" if ( $name1 eq 'chr12:104750000-105500000');
		}
	}
	close(IN);
	close(OUT);
	
	print "bedtools sort -i $tmp_file1 > $outfile\n";
	system("bedtools sort -i $tmp_file1 > $outfile");
	
	print "Write $outfile\n";
}