#!/usr/bin/perl

use strict;
use warnings;
use Getopt::Long;
# keep if enrichment in one side

##############
# Declaration
##############

my $indir = '';
my $infile_name = 'putative_tads_20190521_d15e10i0';

my ($current_sec,$current_min,$current_hour,$current_mday,$current_mon,$current_year,$current_wday,$current_yday,$current_isdst) = localtime();
$current_mon ++;
$current_mon = '0'.$current_mon  if ($current_mon < 10);
$current_mday = '0'.$current_mday  if ($current_mday < 10);
$current_year = $current_year+1900;

##############
# Programs
##############

my $result =  GetOptions ("i=s" => \$indir, "f=s" => \$infile_name);

print "Start $current_mday/$current_mon/$current_year ($current_hour:$current_min)\n";
chdir($indir);
# for (my $i_chr=22; $i_chr <= 22; $i_chr ++){
for (my $i_chr=1; $i_chr <= 24; $i_chr ++){
	my $chr = 'chr'.$i_chr;
	my %Res = ();
	$chr = 'chrX' if ($i_chr == 23);
	$chr = 'chrY' if ($i_chr == 24);
	my @Infile = glob("$infile_name\_part*_$chr\_enrichment.out");
	my $outfile = $indir.$infile_name.'_'.$chr.'_enrichment.out';
	die ("outfile already exists $outfile\n") if (-e $outfile);
	open OUT, '>'.$outfile;	
	foreach my $infile (@Infile){
		print "Read $infile\n";
		my @Header = ();
		open IN,$infile;
		while (<IN>){
			chomp($_);
			if ($#Header < 0){
				@Header = split(/\t/,$_);
				print OUT "$_\n" if (keys(%Res) == 0);
			}
			else{
				my %Column = ();
				readLine($_, \@Header, \%Column);
				my $tad_name = $Column{'Start'}.'-'.$Column{'End'};
				if (exists($Res{$tad_name})){
					die("Different results for $chr:$tad_name\n$_\n") if ($Res{$tad_name} ne $_);
				}
				else{
					print OUT "$_\n";
					$Res{$tad_name} = $_;
				}
			}
		}
		close(IN);
	}
	close(OUT);
	print "Write $outfile\n";
}

($current_sec,$current_min,$current_hour,$current_mday,$current_mon,$current_year,$current_wday,$current_yday,$current_isdst) = localtime();
print "End $current_mday/$current_mon/$current_year ($current_hour:$current_min)\n";

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
