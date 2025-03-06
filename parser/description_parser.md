# Parser Description
This solution made for `theplace.ru` website, which collects high-resolution photos of people. I chose it because of the static and ease of parsing.

## Run
This version doesn't unsupported CLI, you need to change the startup parameters inside `parser.py`.

Parameters list:
- start_url: начальный URL (например, https://www.theplace.ru/photos/)
- min_images (int): requirement to have at least *min_images* photos on the page;
- max_images: will be downloaded max_images for every person;
- max_persons (int): will be made a collection of photos for *max_persons* persons;
- delay (int or float): delay in seconds between requests;
- user_agent: if you need to bypass.

## Save location
A separate folder will be created for each person. For convenience, the author advises to run rename.py to rename photos. 

## TODO
1. Add automatic upload to Google Drive;
2. Add an option for selecting a save location;
3. Add a user-friendly CLI.
