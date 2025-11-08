#!/usr/bin/perl

use strict;
use warnings;
use Getopt::Long;

# keep if enrichment in one side

##############
# Declaration
##############

my $tmpdir = './';
my $infile_name = '';
my $outfile = '';


my %Color = ();
$Color{1} = '153,204,255'; #light blue
$Color{2} = '0,0,255'; # blue
my @Header = ();

my $resolution = 5000;
my $delta_max = 10;
my $delta_diagonale = 15; # warning, : filter alreadt exisy in derivative ==> 10
my $min_reads_per_bin = 2;
my @Value = ();


##############
# Programs
##############


# filter uniquement sur intrac chr
# suprimer very closed
# transformer count
my $result =  GetOptions ("i=s" => \$infile_name, "o=s" => \$outfile, "d=s" => \$delta_diagonale, "e=s" => \$delta_max, "m=s" => \$min_reads_per_bin, "r=s" => \$resolution);

if (exists($ENV{'TMPDIR'})){
    $tmpdir = $ENV{'TMPDIR'};
}
$outfile = $outfile.'_r'.$resolution.'_d'.$delta_diagonale.'_e'.$delta_max.'_m'.$min_reads_per_bin.'.bed';

@Header = ();
open OUT,">".$outfile;

# for (my $i_chr=19; $i_chr <= 19; $i_chr ++){
for (my $i_chr=1; $i_chr <= 24; $i_chr ++){
	my $chr = 'chr'.$i_chr;
	$chr = 'chrX' if ($i_chr == 23);
	$chr = 'chrY' if ($i_chr == 24);
	my $infile = $infile_name.'_'.$chr.'_derivative.out';
	if (-e $infile){
		my %TAD = ();
		my %Coverage = ();
		my %Enrichment = ();
		print "read $infile\n";
		@Header = ();
		open IN,$infile;
		while (<IN>){
			chomp($_);
			if ($#Header < 0){
				$_ =~ s/Deriv_m/Deriv_-/g;
				$_ =~ s/Deriv_p/Deriv_/g;
				$_ =~ s/Cov_m/Cov_-/g;
				$_ =~ s/Cov_p/Cov_/g;
				@Header = split(/\t/,$_);
				# print LOG "$_\tMin\tFilter\n";
			}
			else{
				my %Column = ();
				readLine($_, \@Header, \%Column);
				if (($Column{'End'} - $Column{'Start'} >= $delta_diagonale*$resolution) && ($Column{'Deriv_0'} ne 'NA')) {

					$Enrichment{$Column{'Start'}}{$Column{'End'}}{$Column{'Line'}}{0} = 'NA';
					$Enrichment{$Column{'Start'}}{$Column{'End'}}{$Column{'Line'}}{-1} = 'NA';
					$Enrichment{$Column{'Start'}}{$Column{'End'}}{$Column{'Line'}}{1} = 'NA';
					
					$Enrichment{$Column{'Start'}}{$Column{'End'}}{$Column{'Line'}}{0} = $Column{'Cov_0'} - $Column{'Cov_-1'} if ($Column{'Cov_-1'} ne 'NA');
					$Enrichment{$Column{'Start'}}{$Column{'End'}}{$Column{'Line'}}{-1} = $Column{'Cov_-1'} - $Column{'Cov_-'.$delta_max} if (($Column{'Cov_-1'} ne 'NA') && ($Column{'Cov_-'.$delta_max} ne 'NA'));
					$Enrichment{$Column{'Start'}}{$Column{'End'}}{$Column{'Line'}}{1} = $Column{'Cov_'.$delta_max} - $Column{'Cov_1'} if (($Column{'Cov_'.$delta_max} ne 'NA') && ($Column{'Cov_1'} ne 'NA'));
					$Column{'Deriv_-1'} = 0 if ($Column{'Deriv_-1'} eq 'NA');
					$Column{'Deriv_1'} = 0 if ($Column{'Deriv_1'} eq 'NA');
					
					$Coverage{$Column{'Start'}}{$Column{'End'}}{$Column{'Line'}} = sprintf("%.1f",$Column{'Reads_per_bin'});
					my @Value = ($Column{'Deriv_-1'},$Column{'Deriv_0'},$Column{'Deriv_1'});
					# si <, na car proche de la fenetre max
					# if (($Column{'Cov_-'.$delta_max} ne 'NA') && (($Column{'Cov_0'}-$Column{'Cov_-'.$delta_max}) >= $min_reads_per_bin*$delta_max)){
					# if ($Column{'Deriv_0'} < 0){
					if (($Column{'Deriv_0'} < 0) || (min(\@Value) < -10)){
						# if (($Column{'Cov_-'.$delta_max} ne 'NA') && (($Column{'Cov_0'}-$Column{'Cov_-'.$delta_max}) > $delta_max/2)){
						if (($Column{'Cov_-'.$delta_max} ne 'NA') && (($Column{'Cov_0'}-$Column{'Cov_-'.$delta_max}) >= $delta_max)){
							$TAD{$Column{'Chr'}}{$Column{'Start'}}{$Column{'End'}}{$Column{'Line'}}= min(\@Value) ;
							# print "$Column{'Line'}\t$Column{'Start'}\t$Column{'End'}\t".$Column{'Deriv_0'}."\n" if ($Column{'Start'} == 49565000);
						}
					}
				}
			}
		}
		close(IN);

		my @Start = sort { $a <=> $b} keys(%{$TAD{$chr}});
		foreach my $start (@Start){
			my @End = sort { $a <=> $b} keys(%{$TAD{$chr}{$start}});
			foreach my $end (@End){
				my @Enriched = ();
				push(@Enriched, 'X')  if (exists($TAD{$chr}{$start}{$end}{'X'}) && ($Coverage{$start}{$end}{'X'} >= $min_reads_per_bin));
				push(@Enriched, 'Y') if (exists($TAD{$chr}{$start}{$end}{'Y'}) && ($Coverage{$start}{$end}{'Y'} >= $min_reads_per_bin));
				if ($#Enriched >= 0){
					# just < 0 if X & Y
					# < -10 if only one
					if (($#Enriched == 1) || 
					(($Enriched[0] eq 'X') && ($Enrichment{$start}{$end}{'X'}{1} ne 'NA') && ($Enrichment{$start}{$end}{'X'}{-1} > 1.5*$Enrichment{$start}{$end}{'X'}{1}) && ($Enrichment{$start}{$end}{'X'}{-1}/$delta_max>=$min_reads_per_bin) && ($TAD{$chr}{$start}{$end}{'X'} < -10)) || 
					(($Enriched[0] eq 'Y') && ($Enrichment{$start}{$end}{'Y'}{1} ne 'NA') && ($Enrichment{$start}{$end}{'Y'}{-1} > 1.5*$Enrichment{$start}{$end}{'Y'}{1})&& ($Enrichment{$start}{$end}{'Y'}{-1}/$delta_max>=$min_reads_per_bin) && ($TAD{$chr}{$start}{$end}{'Y'} < -10))){
						print OUT "$chr\t$start\t".($end+$resolution)."\t$chr:$start-".($end+$resolution)."\t".$Coverage{$start}{$end}{'X'}."\t".$Coverage{$start}{$end}{'Y'}."\t$Enrichment{$start}{$end}{'X'}{-1}\t$Enrichment{$start}{$end}{'X'}{0}\t$Enrichment{$start}{$end}{'X'}{1}\t$Enrichment{$start}{$end}{'Y'}{-1}\t$Enrichment{$start}{$end}{'Y'}{0}\t$Enrichment{$start}{$end}{'Y'}{1}\t".join(',',sort(@Enriched))."\n";
					}
				}
			}
		}
	}
	else{
		print STDERR "Warning : No file for $chr : $infile\n";
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
