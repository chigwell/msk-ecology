<?php
global $s;
ini_set("allow_url_fopen", true);
ini_set("display_errors", true);


# Include Instant API's function library.
require_once('class.csv-to-api.php');

# No Source file is given, just show documentation
if ( !isset( $_REQUEST['s'] ) ) {
  die();
}

switch ($_REQUEST['s']) {
    case "1":
        $s  = 'http://msk-ecology.ru/data/traffic_day_dencity.csv';
        break;
    case "2":
        $s = 'http://msk-ecology.ru/data/wind.csv';
        break;
}

# Create a new instance of the Instant API class.
$api = new CSV_To_API();

# Intercept the requested URL and use the parameters within it to determine what data to respond with.
$api->parse_query();

# Gather the requested data from its CSV source, converting it into JSON, XML, or HTML.
$api->parse();

# Send the JSON to the browser.
echo $api->output();