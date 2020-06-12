<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" 
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:template match="/">
        <html>
            <body>
                <h1>NSW and ACT Weather</h1>
                <p>Forecast for the rest of <xsl:value-of select="ms:format-date(product/forecast/area/forecast-period/@start-time-local,'dddd d MMMM')"/>
                </p>
                <p>
                    <xsl:value-of select="product/forecast/area/forecast-period/text"/>
                </p>
            </body>
        </html>
    </xsl:template>
</xsl:stylesheet>