#include "fileoperation.h"
#include <boost/system/error_code.hpp>
#include <boost/asio.hpp>

bool IsFileExistent(const boost::filesystem::path& path) 
{

	boost::system::error_code error;
	auto file_status = boost::filesystem::status(path, error);
	if (error) {
		return false;
	}

	if (!boost::filesystem::exists(file_status)) {
		return false;
	}

	if (boost::filesystem::is_directory(file_status)) {
		return false;
	}

	return true;
}