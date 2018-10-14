library(readr)
library(ggplot2)
library(reshape2)
library(RColorBrewer)
library(extrafont)
loadfonts()


ft_LOO = read_csv("freethrow_computations.csv")
ft_LOO = ft_LOO[order(ft_LOO$n,decreasing=T),] 
ft_LOO = subset(ft_LOO, season %in% c('2015 - 2016', '2014 - 2015', '2013 - 2014') & n > 82)
ft_LOO = ft_LOO[ft_LOO[,4]<0,]
ft_LOO_long = melt(ft_LOO[,-c(1:3,5)],id.vars=c("player","season","n","overall"))

pdf(file = "ft_LOO.pdf", family="CM Roman",width=8,height=11)
base_size = 9
ggplot(ft_LOO_long,aes(x=variable,y=player)) + theme_grey(base_size = base_size) + 
  labs(fill = "LOO(h) - LOO(h=0)") +  labs(x = "",y = "") +
  scale_x_discrete(expand = c(0, 0),labels= c(expression(h=1),expression(h=2),expression(h=3),expression(h=4))
                   ) + scale_y_discrete(expand = c(0, 0)) + 
  geom_tile(aes(fill = value),colour='white') + theme(legend.position="bottom") +
  scale_fill_gradient2(low="olivedrab4", high="coral", guide="colorbar") +
  facet_wrap(~ season, scales = "free") 
dev.off()
embed_fonts("ft_LOO.pdf", outfile="ft_LOO_embed.pdf")
#


ft_percentages_long = melt(ft_LOO[,-c(4:8)],id.vars=c("player","season","n"))
pdf(file = "ft_percentages.pdf", family="CM Roman",width=8,height=11)
base_size = 9
ggplot(ft_percentages_long,aes(x=variable,y=player)) + theme_grey(base_size = base_size) + 
  labs(fill = "probability") +  labs(x = "",y = "") +
  scale_x_discrete(expand = c(0, 0)) + scale_y_discrete(expand = c(0, 0)) + 
  geom_tile(aes(fill = value),colour='white') + geom_text(aes(label = round(value, 2)),size=3) +
  theme(legend.position="bottom") +
  #scale_fill_gradient2(low="olivedrab4", high="coral", guide="colorbar") +
  scale_fill_gradientn(colours=rev(brewer.pal(10,"Spectral"))) + 
  theme(axis.text.x=element_text(angle=33,hjust=1)) +
  facet_wrap(~ season, scales = "free") 
dev.off()
embed_fonts("ft_percentages.pdf", outfile="ft_percentages_embed.pdf")
#
