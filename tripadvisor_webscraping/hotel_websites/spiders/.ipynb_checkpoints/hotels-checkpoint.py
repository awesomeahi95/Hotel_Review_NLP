# -*- coding: utf-8 -*-
import scrapy


class HotelsSpider(scrapy.Spider):
    name = 'hotels'
    allowed_domains = ['www.tripadvisor.co.uk']
    start_urls = ['https://www.tripadvisor.co.uk/Hotel_Review-g187051-d239658-Reviews-Hotel_Hilton_London_Gatwick_Airport-Crawley_West_Sussex_England.html/',
                  'https://www.tripadvisor.co.uk/Hotel_Review-g186338-d193089-Reviews-Hilton_London_Metropole-London_England.html/',
                  'https://www.tripadvisor.co.uk/Hotel_Review-g186338-d192048-Reviews-Hilton_London_Euston-London_England.html/',
                  'https://www.tripadvisor.co.uk/Hotel_Review-g186338-d193102-Reviews-DoubleTree_by_Hilton_Hotel_London_West_End-London_England.html/',
                  'https://www.tripadvisor.co.uk/Hotel_Review-g504167-d192599-Reviews-Hilton_London_Croydon-Croydon_Greater_London_England.html']

    def parse(self, response):
        for review in response.xpath('//div[@class="hotels-community-tab-common-Card__card--ihfZB hotels-community-tab-common-Card__section--4r93H"]'):
            yield {
                'hotel_name': review.xpath('//div[@class="ui_column is-12-tablet is-9-mobile hotels-hotel-review-atf-info-parts-ATFInfo__description--1njly"]/div[1]/h1/text()').get(),
                'review_summary': review.xpath('.//a[@class="location-review-review-list-parts-ReviewTitle__reviewTitleText--2tFRT"]/span[1]/span/text()').get(),
                'review_p1': review.xpath('.//q[@class="location-review-review-list-parts-ExpandableReview__reviewText--gOmRC"]/span[1]/text()').get(),
                'review_p2': review.xpath('.//q[@class="location-review-review-list-parts-ExpandableReview__reviewText--gOmRC"]/span[2]/text()').get(),
                'score': review.xpath('.//div[@class="location-review-review-list-parts-RatingLine__bubbles--GcJvM"]/span').get(),
            }
            
        href = response.xpath('//a[@class="ui_button nav next primary "]/@href').get()
        
        next_page_url = 'https://www.tripadvisor.co.uk' + href
        
        yield scrapy.Request(url=next_page_url, callback = self.parse)
        
        